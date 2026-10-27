# Phase 2b results summary (Qwen2.5-1.5B, Kaggle Tesla T4)

Transcribed from screenshots of the notebook output. The raw `.jsonl` results file was lost when the Kaggle session reset, so these tables are the record. Numbers marked *estimate* are my back-of-envelope calculations, not measurements.

## Setup checks (passed)
- Per-tensor overrides apply: `token_embd.weight -> Q3_K`, `blk.9.attn_v.weight -> Q5_K`, all other weight tensors `Q8_0`.
- Size model: fixed (non-group) bytes 6.5 MB; predicted 1498.7 MB vs actual 1498.7 MB (error +0.00%).
- Cost: about 24-25 s per damage probe (30 chunks); 47-163 s per full test evaluation including `llama-bench`.

## Baselines (test text, 100 chunks)
| Preset | Size (MB) | ppl, no imatrix | ppl, imatrix |
|---|---|---|---|
| F16 | 3094 | 9.6817 | - |
| Q8_0 | 1647 | 9.7018 | - |
| Q6_K | 1273 | 9.7616 | 9.7425 |
| Q5_K_M | 1125 | 9.8090 | 9.7894 |
| Q4_K_M | 986 | 10.0451 | 9.9820 |
| Q3_K_M | 824 | 11.3174 | 10.7184 |
| Q2_K | 676 | 22.7203 | 12.8939 |

The imatrix helps little at 5-6 bits, but a lot at low bits (Q3_K_M -5.3%, Q2_K -43%).

## Optimizer result (verified on the test text)
| Candidate | Size (MB) | ppl | Uniform imatrix ppl at that size | Optimizer vs uniform | GPU tok/s |
|---|---|---|---|---|---|
| opt_824 | 818.8 | 10.730 | 10.791 | +0.6% (noise level) | 142.7 |
| opt_905 | 900.4 | 10.664 | 10.365 | -2.8% | 136.9 |
| opt_986 | 980.4 | 10.606 | 10.007 | -5.6% | 132.4 |
| opt_1056 | 1050.1 | 10.540 | 9.893 | -6.1% | 126.8 |
| opt_1125 | 1118.3 | 10.456 | 9.799 | -6.3% | 122.0 |
| opt_1199 | 1192.7 | 10.357 | 9.768 | -5.7% | 118.8 |
| opt_1273 | 1264.8 | 10.276 | 9.745 | -5.2% | 115.2 |

Negative = optimizer worse. It loses to plain Q6_K at equal size. Actual damage vs q8_0 was 0.057 to 0.101 nats, while predicted damage was about 0.000 for every candidate.

## Sensitivity findings (single tensor at q3_k, everything else q8_0, 30 chunks)
| Kind | Tensors | Mean ppl increase (%) | Max (%) |
|---|---|---|---|
| token_embd | 1 | 15.952 | 15.952 |
| attn_output | 28 | 0.108 | 0.958 |
| ffn_up | 28 | 0.104 | 0.618 |
| ffn_down | 28 | 0.085 | 0.890 |
| ffn_gate | 28 | 0.057 | 0.400 |
| attn_q | 28 | 0.026 | 0.163 |
| attn_v | 28 | 0.026 | 0.180 |
| attn_k | 28 | 0.015 | 0.269 |

Most sensitive tensors: `token_embd` (15.95%, and it holds 233M of about 1.54B weights), `blk.0.attn_output` (0.96%), `blk.27.ffn_down` (0.89%), `blk.27.ffn_up` (0.62%), `blk.1.attn_output` (0.62%), `blk.27.ffn_gate` (0.40%).
Pattern in the heatmap: sensitivity sits at the edges (first blocks' attention output, last block's FFN); middle blocks are near zero. Many tolerant tensors read exactly 0.0 because negative values were floored at zero, so differences below about 0.1% are noise.

## Diagnosis of the optimizer failure
Evidence from the output:
1. **Fitted damage ratios collapsed to zero.** Fitted q4_k / q5_k / q6_k ratios were 0.0000 / 0.0000 / 0.0000, versus 0.2293 / 0.0573 / 0.0131 from the 6-dB-per-bit law.
2. **The probes at q4_k and up are noise.** Validation perplexities scattered between about 8.434 and 8.456 around the 8.4464 reference, with no trend by bit width; some q4_k runs scored *better* than q8_0.
3. **The optimizer therefore saw no cost** in q4_k or lower-damage tensors: predicted damage was 0.0000 at every budget, and no tensor was ever put at q5_k or q6_k (counts: 0 and 0 at most budgets).
4. **The embedding was probably left at q4_k.** At the middle budget the mean bits per weight for `token_embd` was 4.50 (q4_k), while every other kind was about 5.4-5.7.
   *Estimate:* applying the 6-dB law to the embedding's q3_k damage (0.148 nats) gives about 0.034 nats at q4_k, which would explain roughly 60% of the 0.057 nats actually seen in the largest candidate. This is a hypothesis, not a measurement.
5. **MILP was not used.** SciPy failed with `numpy._core._multiarray_umath has no attribute _blas_supports_fpe`, so the greedy fallback ran (likely a numpy/scipy mismatch after a mid-session `pip install`). With all-zero damages the choice was effectively arbitrary anyway.

## Conclusions
- The pipeline works (overrides, size model, imatrix, benchmarking).
- Single-tensor probes cannot resolve the damage of most tensors (below about 0.1%), so a damage model built from them gave no usable guidance.
- Real information exists in a few tensors, mainly the embedding and the edge blocks.
- Phase 3 should search a small number of variables (embedding level, edge-block levels, per-kind levels for the middle blocks) using real whole-model evaluations, starting from the imatrix presets.
