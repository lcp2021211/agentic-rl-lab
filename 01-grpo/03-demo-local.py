"""在本机模型上运行与 PyTRIO demo 相同的 GRPO 训练流程。

与 01-demo-sync.py 相比，数据、prompt、reward、group-relative advantage、
loss 选择和日志指标保持一致；模型加载、rollout、反向传播和 LoRA 保存改为
由本地 Transformers + PEFT 完成。

最小试跑（base-model 必须是本地完整模型目录）：

python 01-grpo/03-demo-local.py \
    --base-model /path/to/Qwen3.5-4B \
    --steps 1 \
    --batch-size 2 \
    --group-size 2 \
    --max-tokens 64 \
    --loss-fn importance_sampling \
    --swanlab-mode disabled
"""

import argparse
import importlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


QUESTION_SUFFIX = " Provide a numerical answer without units, written inside \\boxed{}."
LOSS_FNS = ("importance_sampling", "ppo", "cispo")
PPO_CLIP_EPSILON = 0.2
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    # Qwen3.5 linear-attention blocks.
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    # Qwen3-Next uses a combined b/a projection.
    "in_proj_ba",
    "out_proj",
]
FEWSHOT_PREFIX = [
    {"role": "user", "content": "How many r's are in strawberry?" + QUESTION_SUFFIX},
    {
        "role": "assistant",
        "content": (
            "<think>\n\n</think>\n\n"
            "Let's spell the word out and number all the letters: "
            "1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. "
            "We have r's at positions 3, 8, and 9. "
            "There are three r's. \\boxed{3}"
        ),
    },
]


@dataclass
class GRPOConfig:
    """命令行参数解析后的本地训练配置。"""

    base_model: Path
    output_dir: Path
    lora_rank: int
    steps: int
    all_data: bool
    batch_size: int
    group_size: int
    max_tokens: int
    temperature: float
    top_p: float
    seed: int
    learning_rate: float
    beta1: float
    beta2: float
    loss_fn: str
    cispo_clip_low_threshold: float
    cispo_clip_high_threshold: float
    swanlab_mode: str
    swanlab_project: str


@dataclass
class RolloutSample:
    """本地策略生成的一条 completion 及其采样时 logprob。"""

    tokens: list[int]
    logprobs: list[float]
    text: str
    reward: float
    advantage: float


def parse_args() -> GRPOConfig:
    parser = argparse.ArgumentParser(description="本地模型同步版 GRPO / GSM8K demo")
    parser.add_argument(
        "--base-model",
        type=Path,
        required=True,
        help="本地 Hugging Face 模型目录；不会从 Hub 下载模型",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="最终 LoRA adapter 的父目录",
    )
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank")
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="GRPO 优化步数；每步从 GSM8K 取 batch-size 道题做 rollout",
    )
    parser.add_argument(
        "--all-data",
        action="store_true",
        help="使用 GSM8K train split 全量数据训练一遍；打开后忽略 --steps",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="每个 step 的 GSM8K 题目数")
    parser.add_argument("--group-size", type=int, default=4, help="每道题采样的 completion 数")
    parser.add_argument("--max-tokens", type=int, default=1024, help="每次采样最多生成 token 数")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样 temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="采样 top_p")
    parser.add_argument("--seed", type=int, default=42, help="本地随机种子")
    parser.add_argument("--learning-rate", type=float, default=4e-5, help="Adam learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.95, help="Adam beta2")
    parser.add_argument(
        "--loss-fn",
        choices=LOSS_FNS,
        default="importance_sampling",
        help="训练 loss：importance_sampling / ppo / cispo",
    )
    parser.add_argument(
        "--cispo-clip-low-threshold",
        type=float,
        default=0.0,
        help="CISPO ratio 下界",
    )
    parser.add_argument(
        "--cispo-clip-high-threshold",
        type=float,
        default=4.0,
        help="CISPO ratio 上界",
    )
    parser.add_argument(
        "--swanlab-mode",
        choices=("online", "disabled"),
        default="online",
        help="SwanLab 记录模式",
    )
    parser.add_argument("--swanlab-project", default="agentic-rl-lab-grpo", help="SwanLab project")
    args = parser.parse_args()

    if args.lora_rank <= 0:
        raise ValueError("--lora-rank must be > 0")
    if args.steps <= 0:
        raise ValueError("--steps must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.group_size <= 1:
        raise ValueError("--group-size must be > 1 for group-relative advantage")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be > 0")
    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")
    if args.cispo_clip_low_threshold < 0:
        raise ValueError("--cispo-clip-low-threshold must be >= 0")
    if args.cispo_clip_high_threshold <= 0:
        raise ValueError("--cispo-clip-high-threshold must be > 0")
    if args.cispo_clip_low_threshold > args.cispo_clip_high_threshold:
        raise ValueError(
            "--cispo-clip-low-threshold must be <= --cispo-clip-high-threshold"
        )

    return GRPOConfig(
        base_model=args.base_model.expanduser(),
        output_dir=args.output_dir.expanduser(),
        lora_rank=args.lora_rank,
        steps=args.steps,
        all_data=args.all_data,
        batch_size=args.batch_size,
        group_size=args.group_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        loss_fn=args.loss_fn,
        cispo_clip_low_threshold=args.cispo_clip_low_threshold,
        cispo_clip_high_threshold=args.cispo_clip_high_threshold,
        swanlab_mode=args.swanlab_mode,
        swanlab_project=args.swanlab_project,
    )


def extract_boxed(text: str) -> str | None:
    """取最后一个 boxed answer 作为模型最终答案。"""
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if not matches:
        return None
    return matches[-1].strip()


def normalize_answer(text: str) -> str:
    return text.replace(",", "").strip().rstrip(".")


def grade_answer(response: str, ground_truth: str) -> float:
    answer = extract_boxed(response)
    if answer is None:
        return 0.0
    return 1.0 if normalize_answer(answer) == normalize_answer(ground_truth) else 0.0


def extract_gsm8k_answer(answer_text: str) -> str:
    match = re.search(r"####\s*(.+)", answer_text)
    if match is None:
        raise ValueError(f"No GSM8K final answer found: {answer_text!r}")
    return normalize_answer(match.group(1))


def build_prompt(tokenizer: Any, question: str) -> list[int]:
    messages = [
        *FEWSHOT_PREFIX,
        {"role": "user", "content": question + QUESTION_SUFFIX},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not prompt_tokens:
        raise ValueError("Prompt tokens are empty")
    return prompt_tokens


def load_gsm8k_train() -> Any:
    try:
        datasets = importlib.import_module("datasets")
    except ImportError as exc:
        raise RuntimeError(
            "Local GRPO requires datasets; install 01-grpo/requirements-local.txt"
        ) from exc

    dataset = datasets.load_dataset("openai/gsm8k", "main", split="train")
    if not isinstance(dataset, datasets.Dataset):
        raise TypeError(f"Expected Dataset, got {type(dataset)!r}")
    return dataset


def get_num_steps(dataset: Any, config: GRPOConfig) -> int:
    if config.all_data:
        return (len(dataset) + config.batch_size - 1) // config.batch_size
    return config.steps


def model_slug(base_model: Path) -> str:
    name = base_model.resolve().name.lower().replace("qwen3.5", "qwen35")
    return re.sub(r"[^a-z0-9]+", "-", name).strip("-")


def build_run_name(config: GRPOConfig, effective_steps: int) -> str:
    loss_slug = config.loss_fn.replace("_", "-")
    return (
        f"grpo-local-{model_slug(config.base_model)}-gsm8k-"
        f"{loss_slug}-steps{effective_steps}"
    )


def pick_batch(dataset: Any, step: int, batch_size: int, all_data: bool) -> Any:
    start = step * batch_size
    if all_data:
        end = min(start + batch_size, len(dataset))
        indices = list(range(start, end))
    else:
        indices = [(start + offset) % len(dataset) for offset in range(batch_size)]
    return dataset.select(indices)


def choose_device_and_dtype() -> tuple[torch.device, torch.dtype]:
    if not torch.cuda.is_available():
        print("Warning: CUDA is unavailable; local training will run on CPU and be very slow")
        return torch.device("cpu"), torch.float32
    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return device, dtype


def validate_local_model_dir(model_dir: Path) -> None:
    if not model_dir.is_dir():
        raise ValueError(
            f"--base-model must be an existing local model directory: {model_dir}"
        )
    if not (model_dir / "config.json").is_file():
        raise ValueError(f"Local model directory has no config.json: {model_dir}")
    incomplete_files = sorted(model_dir.glob("*.incomplete"))
    if incomplete_files:
        names = ", ".join(path.name for path in incomplete_files[:3])
        raise ValueError(
            f"Local model download is incomplete ({names}); finish downloading "
            "all weight shards before training"
        )


def load_local_lora_model(
    config: GRPOConfig,
) -> tuple[Any, Any, torch.device, torch.dtype]:
    validate_local_model_dir(config.base_model)

    try:
        peft = importlib.import_module("peft")
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise RuntimeError(
            "Local GRPO requires transformers and peft; "
            "install 01-grpo/requirements-local.txt"
        ) from exc

    device, dtype = choose_device_and_dtype()
    local_path = str(config.base_model.resolve())
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        local_path,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define either pad_token_id or eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token

    model_config = transformers.AutoConfig.from_pretrained(
        local_path,
        local_files_only=True,
    )
    auto_model_class = transformers.AutoModelForCausalLM
    if getattr(model_config, "vision_config", None) is not None:
        auto_model_class = transformers.AutoModelForImageTextToText
    model = auto_model_class.from_pretrained(
        local_path,
        config=model_config,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)

    lora_config = peft.LoraConfig(
        task_type=peft.TaskType.CAUSAL_LM,
        r=config.lora_rank,
        lora_alpha=config.lora_rank,
        lora_dropout=0.0,
        bias="none",
        target_modules=LORA_TARGET_MODULES,
    )
    model = peft.get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer, device, dtype


def get_stop_token_ids(tokenizer: Any) -> list[int]:
    candidates: list[int | None] = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
    ]
    invalid_id = getattr(tokenizer, "unk_token_id", None)
    return list(
        dict.fromkeys(
            token_id
            for token_id in candidates
            if token_id is not None and token_id >= 0 and token_id != invalid_id
        )
    )


def trim_completion_tokens(
    tokens: list[int],
    stop_token_ids: set[int],
    pad_token_id: int,
) -> list[int]:
    """保留第一个停止 token，并去掉 generate 为组内对齐追加的 padding。"""
    for index, token_id in enumerate(tokens):
        if token_id in stop_token_ids:
            return tokens[: index + 1]
    while tokens and tokens[-1] == pad_token_id:
        tokens.pop()
    return tokens


def completion_logprobs(
    model: Any,
    prompt_tokens: list[int],
    completion_tokens: list[int],
    device: torch.device,
) -> torch.Tensor:
    """计算 raw policy 对 completion token 的逐 token logprob。"""
    if not prompt_tokens or not completion_tokens:
        raise ValueError("Prompt and completion must both contain at least one token")
    full_tokens = prompt_tokens + completion_tokens
    input_ids = torch.tensor([full_tokens], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    start = len(prompt_tokens) - 1
    end = len(full_tokens) - 1
    prediction_logits = outputs.logits[:, start:end, :]
    target_ids = input_ids[:, len(prompt_tokens) :]
    # cross_entropy 直接返回目标 token 的 logprob，避免显式保留一个
    # [sequence_length, vocab_size] 的 float32 log_softmax 张量。
    return -torch.nn.functional.cross_entropy(
        prediction_logits.transpose(1, 2),
        target_ids,
        reduction="none",
    ).squeeze(0).float()


@torch.no_grad()
def run_rollout_group(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompt_tokens: list[int],
    ground_truth: str,
    config: GRPOConfig,
) -> list[RolloutSample]:
    """用当前本地策略采样一组 completion，并计算组内 advantage。"""
    model.eval()
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    stop_token_ids = get_stop_token_ids(tokenizer)
    if not stop_token_ids:
        raise ValueError("Tokenizer does not provide a usable generation stop token")

    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=config.max_tokens,
        do_sample=True,
        temperature=config.temperature,
        top_p=config.top_p,
        num_return_sequences=config.group_size,
        eos_token_id=stop_token_ids,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )

    rewards: list[float] = []
    raw_samples: list[tuple[list[int], list[float], str]] = []
    for sequence in generated:
        completion_tokens = trim_completion_tokens(
            sequence[len(prompt_tokens) :].tolist(),
            set(stop_token_ids),
            tokenizer.pad_token_id,
        )
        if not completion_tokens:
            raise ValueError("Model generated an empty completion")
        text = tokenizer.decode(completion_tokens, skip_special_tokens=True)
        old_logprobs = completion_logprobs(
            model,
            prompt_tokens,
            completion_tokens,
            device,
        )
        reward = grade_answer(text, ground_truth)
        rewards.append(reward)
        raw_samples.append(
            (completion_tokens, old_logprobs.cpu().tolist(), text)
        )

    mean_reward = sum(rewards) / len(rewards)
    return [
        RolloutSample(
            tokens=tokens,
            logprobs=logprobs,
            text=text,
            reward=reward,
            advantage=reward - mean_reward,
        )
        for (tokens, logprobs, text), reward in zip(raw_samples, rewards, strict=True)
    ]


def compute_token_loss(
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantage: float,
    config: GRPOConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回逐 token loss、未裁剪 ratio 和该 loss 使用的 ratio。"""
    advantages = torch.full_like(current_logprobs, advantage)
    ratio = torch.exp(current_logprobs - old_logprobs)

    if config.loss_fn == "importance_sampling":
        used_ratio = ratio
        token_loss = -(used_ratio * advantages)
    elif config.loss_fn == "ppo":
        used_ratio = torch.clamp(
            ratio,
            min=1.0 - PPO_CLIP_EPSILON,
            max=1.0 + PPO_CLIP_EPSILON,
        )
        unclipped_objective = ratio * advantages
        clipped_objective = used_ratio * advantages
        token_loss = -torch.minimum(unclipped_objective, clipped_objective)
    elif config.loss_fn == "cispo":
        used_ratio = torch.clamp(
            ratio,
            min=config.cispo_clip_low_threshold,
            max=config.cispo_clip_high_threshold,
        )
        token_loss = -(used_ratio.detach() * current_logprobs * advantages)
    else:
        raise ValueError(f"Unsupported loss function: {config.loss_fn}")

    return token_loss, ratio, used_ratio


def train_on_rollouts(
    model: Any,
    optimizer: torch.optim.Optimizer,
    training_samples: list[tuple[list[int], RolloutSample]],
    config: GRPOConfig,
    device: torch.device,
) -> dict[str, float]:
    """逐条前向并累积梯度，最后执行一次与原 demo 对齐的 optimizer step。"""
    if not training_samples:
        return {}

    denominator = sum(len(sample.tokens) for _, sample in training_samples)
    if denominator == 0:
        return {}

    optimizer.zero_grad(set_to_none=True)
    model.train()
    loss_sum = 0.0
    train_tokens = 0
    ratios: list[torch.Tensor] = []
    used_ratios: list[torch.Tensor] = []

    for prompt_tokens, sample in training_samples:
        current_logprobs = completion_logprobs(
            model,
            prompt_tokens,
            sample.tokens,
            device,
        )
        old_logprobs = torch.tensor(
            sample.logprobs,
            dtype=torch.float32,
            device=device,
        )
        if current_logprobs.shape != old_logprobs.shape:
            raise ValueError("Current and old policy logprobs must have the same shape")

        token_loss, ratio, used_ratio = compute_token_loss(
            current_logprobs,
            old_logprobs,
            sample.advantage,
            config,
        )
        (token_loss.sum() / denominator).backward()
        loss_sum += float(token_loss.detach().sum().item())

        if sample.advantage != 0.0:
            train_tokens += len(sample.tokens)
            ratios.append(ratio.detach())
            used_ratios.append(used_ratio.detach())

    optimizer.step()
    metrics = {
        "loss_mean": loss_sum / denominator,
        "train_tokens": float(train_tokens),
    }
    if config.loss_fn == "cispo" and ratios:
        all_ratios = torch.cat(ratios)
        all_used_ratios = torch.cat(used_ratios)
        metrics.update(
            {
                "cispo/train_tokens": float(train_tokens),
                "cispo/clip_low_threshold": config.cispo_clip_low_threshold,
                "cispo/clip_high_threshold": config.cispo_clip_high_threshold,
                "cispo/ratio_mean": float(all_ratios.mean().item()),
                "cispo/clipped_ratio_mean": float(all_used_ratios.mean().item()),
                "cispo/clip_fraction": float(
                    (all_ratios != all_used_ratios).float().mean().item()
                ),
            }
        )
    return metrics


def init_swanlab_run(
    config: GRPOConfig,
    effective_steps: int,
    dataset_size: int,
    run_name: str,
) -> tuple[Any, Any] | None:
    if config.swanlab_mode == "disabled":
        return None
    try:
        swanlab = importlib.import_module("swanlab")
    except ImportError as exc:
        raise RuntimeError(
            "SwanLab online logging requires swanlab; "
            "install 01-grpo/requirements-local.txt or use --swanlab-mode disabled"
        ) from exc

    run = swanlab.init(
        project=config.swanlab_project,
        name=run_name,
        mode=config.swanlab_mode,
        config={
            "base_model": str(config.base_model.resolve()),
            "output_dir": str(config.output_dir.resolve()),
            "run_name": run_name,
            "lora_rank": config.lora_rank,
            "steps": config.steps,
            "all_data": config.all_data,
            "effective_steps": effective_steps,
            "batch_size": config.batch_size,
            "dataset_size": dataset_size,
            "group_size": config.group_size,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "learning_rate": config.learning_rate,
            "beta1": config.beta1,
            "beta2": config.beta2,
            "loss_fn": config.loss_fn,
            "ppo_clip_epsilon": PPO_CLIP_EPSILON,
            "cispo_clip_low_threshold": config.cispo_clip_low_threshold,
            "cispo_clip_high_threshold": config.cispo_clip_high_threshold,
            "swanlab_mode": config.swanlab_mode,
            "seed": config.seed,
            "weights_name": run_name,
        },
    )
    return swanlab, run


def main(config: GRPOConfig) -> None:
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    validate_local_model_dir(config.base_model)
    print("Loading GSM8K dataset...")
    train_data = load_gsm8k_train()
    print(f"Loaded {len(train_data)} GSM8K training examples")
    effective_steps = get_num_steps(train_data, config)
    run_name = build_run_name(config, effective_steps)
    print(f"Run / weights name: {run_name}")
    if config.all_data:
        print(
            f"All-data mode: {effective_steps} steps will cover "
            f"{len(train_data)} examples once"
        )

    print(f"Loading local model from {config.base_model.resolve()}...")
    model, tokenizer, device, dtype = load_local_lora_model(config)
    print(f"Local training device: {device}, dtype: {dtype}")
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )
    swanlab_state = init_swanlab_run(
        config=config,
        effective_steps=effective_steps,
        dataset_size=len(train_data),
        run_name=run_name,
    )

    try:
        for step in range(effective_steps):
            batch_rows = pick_batch(
                train_data,
                step,
                config.batch_size,
                config.all_data,
            )
            training_samples: list[tuple[list[int], RolloutSample]] = []
            prompt_mean_rewards: list[float] = []
            rollout_lengths: list[int] = []
            n_degenerate = 0

            for row in tqdm(batch_rows, desc=f"GRPO step {step}", unit="prompt"):
                prompt_tokens = build_prompt(tokenizer, row["question"])
                ground_truth = extract_gsm8k_answer(row["answer"])
                rollout_samples = run_rollout_group(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    prompt_tokens=prompt_tokens,
                    ground_truth=ground_truth,
                    config=config,
                )

                rollout_lengths.extend(len(sample.tokens) for sample in rollout_samples)
                rewards = [sample.reward for sample in rollout_samples]
                prompt_mean_rewards.append(sum(rewards) / len(rewards))
                if all(sample.advantage == 0.0 for sample in rollout_samples):
                    n_degenerate += 1
                    continue
                training_samples.extend(
                    (prompt_tokens, sample) for sample in rollout_samples
                )

            loss_metrics = train_on_rollouts(
                model=model,
                optimizer=optimizer,
                training_samples=training_samples,
                config=config,
                device=device,
            )
            loss_mean = loss_metrics.get("loss_mean")
            train_tokens = int(loss_metrics.get("train_tokens", 0.0))
            mean_reward = sum(prompt_mean_rewards) / len(prompt_mean_rewards)
            avg_gen_len = (
                sum(rollout_lengths) / len(rollout_lengths)
                if rollout_lengths
                else 0.0
            )
            frac_degenerate = n_degenerate / len(prompt_mean_rewards)

            if swanlab_state is not None:
                swanlab, _ = swanlab_state
                log_payload = {
                    "reward": mean_reward,
                    "frac_degenerate": frac_degenerate,
                    "rollout/avg_gen_len": avg_gen_len,
                    "datums": len(training_samples),
                    "train_tokens": train_tokens,
                    **{
                        key if key.startswith("cispo/") else f"trainer/{key}": value
                        for key, value in loss_metrics.items()
                        if key == "loss_mean" or key.startswith("cispo/")
                    },
                }
                if loss_mean is not None:
                    log_payload["loss"] = loss_mean
                    log_payload["loss_mean"] = loss_mean
                swanlab.log(log_payload, step=step)

            loss_mean_text = "n/a" if loss_mean is None else f"{loss_mean:.4f}"
            print(
                f"Step {step:2d} | reward: {mean_reward:.3f} | "
                f"degenerate: {frac_degenerate:.0%} | "
                f"datums: {len(training_samples)} | train_tokens: {train_tokens} | "
                f"avg_gen_len: {avg_gen_len:.1f} | "
                f"loss_mean: {loss_mean_text} | loss_fn: {config.loss_fn}"
            )

        final_output_dir = config.output_dir.resolve() / run_name
        final_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving final LoRA adapter to {final_output_dir}...")
        model.save_pretrained(final_output_dir, safe_serialization=True)
        tokenizer.save_pretrained(final_output_dir)
        print(f"Saved LoRA adapter: {final_output_dir}")
    finally:
        if swanlab_state is not None:
            _, swanlab_run = swanlab_state
            swanlab_run.finish()


if __name__ == "__main__":
    cli_config = parse_args()
    start_main_time = time.time()
    main(cli_config)
    end_main_time = time.time()
    print("#" * 50)
    print("# all done")
    print(f"# train cost {end_main_time - start_main_time:.2f}s")
    print("#" * 50)
