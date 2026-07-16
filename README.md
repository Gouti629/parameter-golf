# `tied-embeddings` — Tied vs. Untied Embeddings

![Embedding for decoder only architecture](<ChatGPT Image Jul 15, 2026, 02_15_16 PM.png>)

## Motivation

In an LLM, each token starts as nothing but an index or an a discrete ID carrying no notion of similarity to any other token. The embedding matrix maps that index to a dense, continuous vector, and it's this vector that carries meaning.

At the other end of every forward pass, the model has to turn its final hidden state back into a prediction. It does so by projecting that state onto the vocabulary to get one score per token, which softmax turns into a probability distribution.

Both matrices are `vocab_size × model_dim` not merely similar in shape, but identical and
they're doing mirror-image jobs.
Row `i` of the input embedding is "the vector that means token `i`." Row `i` of the output matrix is "the direction a hidden state must point for token `i` to score highly," which is, in effect, also a vector meaning token `i`. Two separately-learned matrices converging on the same kind of knowledge.

The idea of **tied embeddings** is to use one matrix for both jobs i.e, a lookup on the way in, a dot-product scorer on the way out. This reduces the parameter count and also acts as a regularizer, since a token's input and output representations can no longer drift apart.

### Why the practice is scale-dependent

The empirical pattern is that models with tied embeddings tend to be the smaller ones — Qwen2.5 3B, OLMo 1B, Gemma 2 9B, with Gemma 2 27B as a notable counterexample that ties despite its size.
Qwen2.5-Coder's report states the split outright: the 1.5B uses embedding tying, the 7B does not, despite an identical 151,646-token vocabulary.

This makes sense on two counts. On **parameters**, embeddings grow as `V × d` while the transformer body grows as roughly `12 × L × d²`, so the embedding share collapses as models scale:

At 1B, tying is a ~20% parameter cut worth spending on more layers. At 70B it's ~1.5% — a rounding error, so you take the free expressiveness instead. (Gemma 2 27B shows share isn't the only input: its 256k vocab keeps the *absolute* saving large, and Google ties across the whole family.)

On **regularization**, scaling laws mean larger models are trained on correspondingly larger
corpora — modern LLMs see well under one epoch of trillions of tokens, so overfitting is essentially absent and there's nothing for the constraint to regularize away. Whatever overfitting pressure exists lives in the small-model regime, which is the same place the parameter argument bites.

### Where this config actually sits
Worth being explicit about, because it's counterintuitive: at `vocab_size=1024`, the embedding here is **3.1% of the model**  which is in the Llama-70B territory, not Llama-1B territory, despite this being a 17M-parameter model.

So this branch isn't really testing "tying in a small model." It's testing tying at an embedding share where the field has already concluded untying wins.

## What changed?

Nothing structural — the baseline already provides a `TIE_EMBEDDINGS` argument (`1` for tied, `0` for untied, default `1`), so this branch runs both settings rather than implementing anything new.
Attention stays GQA (`num_heads=8`, `num_kv_heads=4`) throughout, unchanged from the baseline and shared across both arms.

But the two arms differ in more than weight sharing, and this matters for interpreting the results.

**Different learning rates.**
| | Input embedding | Output head |
|---|---|---|
| Untied | `EMBED_LR = 0.6` | `HEAD_LR = 0.008` |
| Tied | `TIED_EMBED_LR = 0.05` (both) | — |
A 75× spread, for a reason visible in the code:


(This different Learning Rates piqued my interest and hence I checked the reason with Claude Opus 4.8 High Effort model, The reason below is pasted as is to make sure I do not miss any important points)
On the **input** side the embedding is RMS-normed the instant it's looked up, so only direction matters — magnitude is thrown away. Hence untied can afford `lr=0.6`. On the **output** side, `F.linear(x, W)` feeds straight into a `tanh` logit softcap, where magnitude is everything — too large and tanh saturates, killing the gradient. Hence `HEAD_LR = 0.008`.

Tying forces one tensor to serve a scale-invariant consumer and a scale-critical one at once, so a single LR must satisfy both. `TIED_EMBED_LR = 0.05` is that compromise — closer to the head's 0.008 than the embedding's 0.6, because the output path is the binding constraint. 


## Parameter count

Config: `model_dim=512`, `num_heads=8`, `num_kv_heads=4`, `head_dim=64`, `mlp_mult=2`,
`num_layers=9`, `vocab_size=1024`.

### Per block (identical in both arms)

| Component | Shape | Params |
|---|---|---|
| `c_q` | 512 × 512 | 262,144 |
| `c_k` | 512 × 256 | 131,072 |
| `c_v` | 512 × 256 | 131,072 |
| `proj` | 512 × 512 | 262,144 |
| `q_gain` | 8 | 8 |
| **Attention** | | **786,440** |
| MLP (`fc` + `proj`) | 512×1024 + 1024×512 | 1,048,576 |
| Block scalars (`attn_scale`, `mlp_scale`, `resid_mix`) | | 2,048 |
| **Block total** | | **1,837,064** |

9 blocks = 16,533,576 · `skip_weights` (4 × 512) = 2,048

### The two arms

| | `tok_emb` | `lm_head` | Blocks + skips | **Total** |
|---|---|---|---|---|
| **Tied** | 524,288 | — (shared) | 16,535,624 | **17,059,912** |
| **Untied** | 524,288 | 524,288 | 16,535,624 | **17,584,200** |

Untying costs **524,288 parameters, a 3.07% increase**\

## Setup

Single H100 (Runpod), `sp1024` FineWeb data/tokenizer. The challenge specifies 8×H100; we used one due to budget constraints, so absolute bpb is not comparable to the leaderboard. The comparison between arms remains valid, as both run under identical conditions.

The challenge caps training at 10 minutes of wallclock. Since untied adds a matmul for `lm_head` plus an extra optimizer group, we expected it to complete fewer iterations in the same budget. So, we also ran both arms for a fixed 5000 iterations to separate "better per step" from "more steps."

Three seeds per arm at the 10-minute cap, one seed per arm at 5000 iterations.

## Results

| Config | Embeddings | Seed | Val bpb | Model params | Iterations | Wallclock (min) |
|---|---|---|---|---|---|---|
| `tied_s1` | Tied | 123 | 1.345 | 17,059,912 | 1,153 | 10 |
| `tied_s2` | Tied | 12 | 1.342 | 17,059,912 | 1,170 | 10 |
| `tied_s3` | Tied | 712 | 1.343 | 17,059,912 | 1,173 | 10 |
| `tied_s4` | Tied | 123 | 1.277 | 17,059,912 | 5,000 | 44 |
| `untied_s1` | Untied | 123 | 1.324 | 17,584,200 | 1,163 | 10 |
| `untied_s2` | Untied | 12 | 1.325 | 17,584,200 | 1,170 | 10 |
| `untied_s3` | Untied | 712 | 1.323 | 17,584,200 | 1,170 | 10 |
| `untied_s4` | Untied | 123 | 1.269 | 17,584,200 | 5,000 | 46 |

### Summary (10-minute runs, 3 seeds each)

| Embeddings | Mean bpb | Std | Min | Max |
|---|---|---|---|---|
| Tied | 1.3433 | 0.0015 | 1.342 | 1.345 |
| Untied | **1.3240** | 0.0010 | 1.323 | 1.325 |

**Untied wins cleanly, with no overlap between the arms.** The worst untied run (1.325) beats the best tied run (1.342). A 1.5% mean gap against a ~0.1% noise floor is about as separated as a three-seed result gets.

### The gap shrinks with training

| Iterations | Tied | Untied | Gap |
|---|---|---|---|
| ~1,170 | 1.3433 | 1.3240 | **1.44%** |
| 5,000 | 1.277 | 1.269 | **0.63%** |

Untied's advantage more than halves by 5000 iterations. This is a single seed per arm, so treat it as suggestive rather than established.
This result hints that part of untied's early lead is optimization speed rather than permanent capacity. Untied starts at exactly uniform logits (zero-init head) and lets the input embedding move 12× faster (`EMBED_LR=0.6` vs `TIED_EMBED_LR=0.05`), both of
which should matter most early.

### Step time

The 10-minute runs show no meaningful difference with both tied and untied completing ~1,165 iterations each (≈0.515 s/iter), so the expected untied slowdown didn't materialize at that scale. 
The 5000-iteration runs disagree with tied and untied being 44 min vs 46 min respectively, making untied ~4.5% slower. The two setups contradict each other and neither is profiled, so the honest answer is that any step-time difference is small and we haven't measured it well enough to characterize.

## Takeaways

- **Untied outperforms tied in this configuration** — 1.324 vs 1.343 bpb at the 10-minute cap, with complete separation across three seeds. Not a large margin, but an unambiguous one.
- **The advantage narrows with training** (1.44% → 0.63% by 5000 iterations), suggesting some of it is early-training optimization speed rather than expressiveness. Single seed; needs replication.
- **At 3.1% embedding share, this config sits in the regime where the field already unties** The result reproduces Qwen and Llama's choices at comparable shares rather than contradicting them. The 16 MB cap forced a small vocab, and a small vocab is what put a 17M-parameter model in the large-model regime.

## Open questions / next steps

- Run both arms to convergence the narrowing gap suggests the ordering might not survive.
- Replicate the 5000-iteration runs across seeds; one seed per arm isn't enough to trust the
  narrowing.
- Sweep through multiple LLM sizes to see where tying starts to pay.