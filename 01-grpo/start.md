# GRPO 快速启动

以下命令均从仓库根目录运行。项目要求 Python `>=3.13`。

```bash
uv sync
trio login
swanlab login
```

## 1. 准备数据

无需单独执行下载脚本。训练首次启动时会自动下载 `openai/gsm8k` 的 train split，并使用本地 Hugging Face 缓存。

## 2. 启动训练

当前依赖版本下，直接使用同步入口：

```bash
uv run python 01-grpo/01-demo-sync.py \
    --steps 10 \
    --batch-size 4 \
    --group-size 8 \
    --max-tokens 512 \
    --loss-fn importance_sampling \
    --swanlab-mode online
```

`02-demo-async.py` 仍包含旧版 PyTRIO 的 timeout 配置，更新该参数前会在启动阶段报错，因此快速指南暂不使用它。

### 在本机训练本地模型

`03-demo-local.py` 保留相同的 GSM8K、prompt、reward、group advantage、三种 loss 和训练参数，
但不再连接 PyTRIO；rollout、LoRA 反向传播和权重保存都在本机执行。

请先准备带 CUDA 的 PyTorch 环境，再安装本地入口依赖：

```bash
pip install -r 01-grpo/requirements-local.txt
```

然后传入一个**已经完整下载到本地**的 Hugging Face 模型目录：

```bash
python 01-grpo/03-demo-local.py \
    --base-model /path/to/Qwen3.5-4B \
    --steps 10 \
    --batch-size 4 \
    --group-size 8 \
    --max-tokens 512 \
    --loss-fn importance_sampling \
    --swanlab-mode online
```

低成本连通性测试：

```bash
python 01-grpo/03-demo-local.py \
    --base-model /path/to/Qwen3.5-4B \
    --steps 1 \
    --batch-size 2 \
    --group-size 2 \
    --max-tokens 64 \
    --loss-fn importance_sampling \
    --swanlab-mode disabled
```

模型文件使用 `local_files_only=True` 加载，不会回退到 Hugging Face Hub。最终 LoRA adapter
默认保存到 `01-grpo/outputs/<run-name>/`，也可以通过 `--output-dir` 修改。为了避免一次
forward 占用过多显存，本地入口逐条累积 rollout 梯度，并且仍然只在每个 GRPO step 更新一次。

## 3. 运行评测

当前项目没有独立的 `eval.py`。训练过程会在终端和 SwanLab 中记录 `reward`、`frac_degenerate`、`rollout/avg_gen_len`、`train_tokens` 与 `loss_mean`，并在结束时打印保存的 sampler weights 路径。

比较不同 loss 时，保持其他参数一致，仅修改：

```bash
--loss-fn importance_sampling
--loss-fn ppo
--loss-fn cispo
```
