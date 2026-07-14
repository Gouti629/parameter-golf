# parameter-golf (fork) Pre-training Experiments

This is a personal fork of **openai/parameter-golf**, used as a sandbox for learning and
experimenting with LLM pre-training internals hands-on. Each branch is a self-contained
experiment: a change to `train_gpt.py` (or the training setup), a run against the `sp1024`
FineWeb baseline, and notes on what I learned.

The goal isn't to top the leaderboard but the aim is to build real intuition for how each
architectural or optimization choice actually affects loss, stability, and throughput, by
changing one thing at a time and comparing against the baseline.

Although the challenge specifies 8×H100, these experiments were run on a single H100 (Runpod)
due to compute constraints. The 16 MB model artifact size limit was kept as-is to stay within
the intended budget.

## Branches

| Branch | Experiment | Status | Notes |
|---|---|---|---|
| [`tied-embeddings`](../../tree/tied-embeddings) | Tied vs. untied embeddings | Done | [README](../../tree/tied-embeddings/README.md) |
| [`mla`](../../tree/mla) | Multi-head Latent Attention vs. GQA | Done | [README](../../tree/mla/README.md) |