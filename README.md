# Automated mixed-precision quantization of LLMs for edge deployment

Final-year project. Given a model and a file-size budget, this pipeline decides how many bits (3 to 8) each
tensor family of the model gets, and checks the result against the standard llama.cpp quantization presets.

**Headline results (WikiText-2 test, full split, same size or smaller than the baseline)**

| Model | Size budget | Allocation ppl | Preset / preset curve ppl | Gain |
|---|---|---|---|---|
| Qwen2.5-1.5B | 824 MB (Q3_K_M) | 10.449 | 10.586 | 1.3% |
| Qwen2.5-1.5B | 870 MB (no preset) | 10.140 | 10.320 (curve) | 1.7% |
| Qwen2.5-1.5B | 920 MB (no preset) | 10.019 | 10.105 (curve) | 0.9% |
| Qwen2.5-1.5B | 986 MB (Q4_K_M) | 9.848 | 9.923 | 0.8% |
| SmolLM2-1.7B | 860 MB (Q3_K_M) | 9.340 | 9.409 | 0.7% |
| SmolLM2-1.7B | 925 MB (no preset) | 9.109 | 9.122 (curve) | 0.15% (tie) |
| SmolLM2-1.7B | 990 MB (no preset) | 8.929 | 8.924 (curve) | -0.05% (tie) |
| SmolLM2-1.7B | 1056 MB (Q4_K_M) | 8.776 | 8.802 | 0.3% |

All baselines are llama.cpp presets quantized **with an importance matrix**. "Curve" = monotone interpolation of
log-perplexity through four presets (Q2_K, Q3_K_M, Q4_K_M, Q5_K_M). Random allocations of the same size were
worse in every case (by 1.5% up to 20%).

**Laptop speed (Qwen2.5-1.5B, RTX 4050 and CPU, llama.cpp b11851, `llama-bench`, tokens/s)**

| Model | Size (MB) | Test ppl | GPU generation | CPU generation (6 threads) |
|---|---|---|---|---|
| Q3_K_M preset | 824 | 10.586 | 142.8 | 37.8 |
| allocation @824 | 824 | 10.449 | 178.9 | 38.2 |
| allocation @870 | 870 | 10.140 | 170.8 | 35.3 |
| allocation @920 | 920 | 10.019 | 157.6 | 33.5 |
| Q4_K_M preset | 986 | 9.923 | 149.7 | 31.5 |
| allocation @986 | 986 | 9.848 | 149.4 | 31.7 |

Speed depends on the device: on a Kaggle Tesla T4 the 824 MB allocation was slightly *slower* than Q3_K_M. Always
measure on the target device.

## Method

1. **Baselines.** Quantize presets with an importance matrix; evaluate perplexity on the full WikiText-2 test split.
2. **Tensor families.** Group the 2-D weight matrices by kind (attn_q/k/v/output, ffn_gate/up/down) and position
   (first 2 blocks, middle, last 2 blocks), plus the embedding: 22 families.
3. **Probes.** Set one family at a time to q3_k and q4_k (everything else q8_0) and measure the perplexity increase
   on a *search* text (a train-split slice). Three large families are also probed at q5_k / q6_k to calibrate how
   fast damage shrinks with more bits.
4. **Damage model.** damage(family, level) = scale(family) x r(level), fitted by least squares.
5. **Exact solver.** Dynamic programming (multiple-choice knapsack) picks one level per family under a size budget.
   An exact size model (error 0.00% against real files) is used.
6. **Verification.** Every allocation is quantized and evaluated on the held-out test split; compared with the
   presets, the interpolated preset curve and random allocations of the same size.

## Findings

- The **embedding** is by far the most sensitive tensor: with everything else at q8_0, q4_k costs +3.7% perplexity,
  q5_k +0.8%, q6_k +0.2% (Qwen2.5-1.5B). On SmolLM2 it costs 20% at q3_k although it is only 6% of the weights.
- A first approach based on **single-tensor probes failed**: their effects were below measurement noise, the fitted
  damage at q4_k and above collapsed to zero, and the resulting allocations were 6-10% worse than uniform presets.
  Probing **whole families** fixed it (predicted vs measured damage within about 5-20%).
- Damages of different families are **approximately additive** (joint damage 2-4% above the sum of the parts).
- A measurement-driven hill-climb from the Q4_K_M preset gained 0.35% on the full test split (random proposals
  0.16%) but needed 3.3 GPU hours; the family model needed about 30 minutes of probing.
- The benefit is **model dependent**: 0.8-1.7% on Qwen2.5-1.5B, 0.3-0.7% at preset sizes (ties in between) on SmolLM2-1.7B.

## Repository layout

```
notebooks/
  phase1_baseline_quantization.ipynb     # baselines: F16, Q8_0 ... Q2_K, perplexity, speed
  phase2a_block_sensitivity.ipynb        # first sensitivity attempt (blocks, q8_0/q2_k)
  phase2b_multilevel_optimizer.ipynb     # single-tensor probes + optimizer (negative result)
  phase3_measurement_search.ipynb        # hill-climb from presets, embedding + additivity experiments
  phase2c_family_model_solver.ipynb      # family-level model + exact solver (main result)
  phase4_generalization_smollm2.ipynb    # second model
  rebuild_phase2c_models.ipynb           # rebuilds the benchmarked models
results/
  laptop_benchmarks_summary.csv, bench_results.txt, phase2b_*.csv, ...
docs/figures/                            # plots exported from the notebooks
```

(Adjust the names to match your files.)

## Reproducing

- Notebooks are written for **Kaggle** (Tesla T4, internet on). Each builds llama.cpp from source at commit
  `6b9ba4f`, downloads the model, converts it to GGUF, computes an importance matrix and runs the pipeline.
- Laptop benchmarks use the prebuilt Windows releases of llama.cpp (**b11851**): the CUDA build for the GPU and the
  CPU build with `-ngl 0` for CPU-only numbers.
- Perplexity data: WikiText-2 (raw). Texts are split so that the importance matrix (train, first 3M characters),
  the search text (train, a different slice) and the final numbers (test split) never overlap.
- Models: `Qwen/Qwen2.5-1.5B` and `HuggingFaceTB/SmolLM2-1.7B` (check each model card for its license).

## Limitations

- Two models, one text domain (WikiText-2), perplexity as the main quality metric. Downstream task accuracy
  (HellaSwag, 400 tasks, about +-2.4 points) cannot resolve differences of about 1%.
- Differences of 0.3% are small; paired comparisons on identical text are more precise than the stated absolute
  errors, but a second held-out text would make them firmer.
- Speed was measured on one laptop with synthetic benchmarks (pp512, tg128); peak memory was not measured.
- Allocation is one level per tensor family, so it cannot copy the preset's internal alternating patterns.
- Models whose weight rows are not multiples of 256 (for example Qwen2.5-0.5B, hidden size 896) cannot use K-quants
  as assumed; the notebooks stop with an error in that case.

## Tools

Python, llama.cpp (llama-quantize, llama-imatrix, llama-perplexity, llama-bench), GGUF, CUDA, Hugging Face
(Transformers, Datasets, Hub), NumPy, pandas, Matplotlib, SciPy (interpolation), Kaggle GPUs, PowerShell.

## License

Code: choose a license (for example MIT). Models and datasets keep their own licenses.
