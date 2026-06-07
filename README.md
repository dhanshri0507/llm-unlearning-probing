# LLM Unlearning Probing Experiment
### *Forget-entity knowledge remains linearly decodable from intermediate layer activations after NPO+RT unlearning, even when behavioral metrics indicate forgetting.*

---

## Motivation

**GDPR's Right to be Forgotten** (Article 17) requires that personal data be removed from systems on request, including, arguably, from trained machine learning models. Machine unlearning has emerged as the technical response: methods that modify a model's weights post-training so that specified training examples are no longer recoverable.

The dominant evaluation paradigm checks only *behavioral* forgetting: does the model still output the target information when queried? This experiment tests a complementary question: **even if behavioral output is suppressed, does the knowledge remain encoded in intermediate layer activations in a linearly accessible form?**

This question has direct practical relevance to the **OPT-OUT** framework (ACL 2025, Choi et al.), which proposes a principled unlearning evaluation protocol for language models trained on user data. OPT-OUT distinguishes between surface-level output suppression and deeper representational erasure. The present experiment provides empirical evidence for the gap between these two levels on the TOFU benchmark.

---

## Hypothesis

> After NPO+RT unlearning, forget-entity knowledge remains **linearly decodable** from intermediate transformer layer activations, even when behavioral metrics (ROUGE-L on generation) show the model has forgotten.

---

## Methodology

### Dataset
- **TOFU** (`locuslab/TOFU`, `forget05` split): 200 QA pairs about 10 fictitious authors invented for this benchmark, ensuring no knowledge leakage from real pre-training data.
- Splits: 200 forget QA pairs, 150 retain-train QA pairs, 50 retain-validation QA pairs.

### Model
- **Phi-3.5-mini-instruct** (`microsoft/Phi-3.5-mini-instruct`)
- 4-bit NF4 quantization via `bitsandbytes` + `BitsAndBytesConfig`
- LoRA adapters (rank 8, alpha 16) via `peft` for trainable unlearning on T4

### Unlearning: NPO+RT
**Negative Preference Optimisation + Retain Training** (NPO+RT) is a gradient-based unlearning method that:
1. Applies a contrastive loss to push the model's log-probabilities on forget-set answers *away* from the frozen reference model's predictions.
2. Simultaneously applies a standard cross-entropy retention loss on retain-set examples to prevent catastrophic forgetting.

Training configuration: 100 steps, `lr=2e-6`, `batch_size=1`, `max_length=128`.

> **Note:** GA+RT (Gradient Ascent + Retain Training) was attempted but produced NaN activations from gradient ascent instability on T4 and was excluded. NPO+RT is also the more relevant baseline given its design proximity to OPT-OUT's preferred unlearning objective.

### Activation Extraction
- Hidden states extracted at the **last real token position** (last non-padding token of the input question) across layers **4, 8, 12, 16, 20, 24, 28, 32**.
- Extraction used `output_hidden_states=True` with `torch.no_grad()` and immediate `.detach().cpu()` to avoid gradient accumulation.
- Subsets: 100 forget samples + 100 retain samples (balanced probe dataset).

### Linear Probing
- Pipeline: `SimpleImputer(strategy='median')` → `StandardScaler` → `LogisticRegression(max_iter=1000, C=1.0)`
- Evaluation: **5-fold stratified cross-validation**, metric = mean accuracy
- Binary classification: forget entity (label 1) vs. retain entity (label 0)
- Random chance baseline: 0.500

---

## Results

| Model | Forget ROUGE-L | Retain ROUGE-L | Mean Probe Acc (all layers) | Mean Probe Acc (layers 16–32) | Layer 32 Probe Acc |
|-------|:--------------:|:--------------:|:---------------------------:|:-----------------------------:|:------------------:|
| Original | 0.306 | 0.299 | 0.543 | 0.575 | 0.640 |
| NPO+RT | **0.148** | 0.260 | **0.621** | **0.663** | **0.835** |
| Random chance | — | — | 0.500 | 0.500 | 0.500 |

**Key observations:**
- Forget ROUGE-L drops from 0.306 → 0.148 (51% relative reduction), confirming behavioral suppression.
- Retain ROUGE-L falls modestly from 0.299 → 0.260, indicating retain training successfully limits catastrophic forgetting.
- **Counterintuitively, probe accuracy rises after unlearning** from 0.543 → 0.621 (all layers) and from 0.640 → 0.835 at layer 32.
- This increase is concentrated in deep layers (16–32), suggesting that NPO+RT's weight updates **reorganize and concentrate** forget-entity representations in later layers rather than erasing them.

---

## Figures

### `fig1_probe_accuracy.png` — Layer-wise Probe Accuracy
Line plot showing mean 5-fold CV probe accuracy vs. transformer layer index (4, 8, 12, …, 32) for the Original and NPO+RT models. Both curves are above random chance (0.50 dashed line). The NPO+RT curve crosses above the Original curve in layers 16–32, with the largest divergence at layer 32 (0.640 → 0.835).

### `fig2_pca.png` — PCA of Layer-20 Activations
Side-by-side PCA scatter plots (2 principal components) of layer-20 hidden states, colored by forget (red) vs. retain (blue) entity class. The NPO+RT panel shows visually cleaner cluster separation than the Original panel, consistent with the linear probe increase.

### `fig3_summary.png` — Behavioral vs. Representational Summary
Two-panel bar chart. Left panel: ROUGE-L bars for forget and retain across Original / NPO+RT (behavioral level). Right panel: mean probe accuracy bars for Original / NPO+RT vs. random chance threshold (representational level). Illustrates the dissociation between behavioral suppression and representational persistence.

### `training_curves.png` — NPO vs. RT Loss Components
Dual-axis loss curve over 100 training steps. NPO loss (forget, red) decreases monotonically; RT loss (retain, blue) remains stable. Confirms the unlearning procedure ran correctly without model collapse.

---

## Interpretation

The post-unlearning **increase** in probe accuracy is not predicted by a simple output-suppression account of NPO+RT. A suppression-only mechanism would leave probe accuracy approximately constant (the representations would be unchanged, only the output head would re-weight them).

The observed pattern stable behavioral metrics for retain, reduced behavioral metrics for forget, *but higher* representational separability is consistent with a **knowledge reorganization hypothesis**: NPO+RT's gradient updates push forget-entity representations into a more geometrically distinct subspace in deeper layers, where a linear probe can more easily decode class membership, while the output layers learn to ignore or suppress that subspace during generation.

This has immediate implications for GDPR compliance: a model that passes standard unlearning benchmarks (output-level ROUGE-L) may still be vulnerable to representation-level extraction via activation probing or model internals analysis.

---

## Setup Instructions (Google Colab T4)

1. Open in Google Colab: **Runtime → Change runtime type → T4 GPU**

2. Install dependencies (Cell 1 of the notebook requires runtime restart):

```bash
pip install transformers==4.44.0 accelerate==0.33.0 datasets \
            bitsandbytes peft scikit-learn evaluate rouge_score \
            matplotlib seaborn pandas tqdm
```

3. After the kernel restarts, run all subsequent cells in order.

4. Results (figures + CSV + JSON) are saved to `/content/results/` and downloaded as `unlearning_results.zip`.

**Estimated runtime:** ~45–75 minutes on T4 (model load + 100 unlearning steps + activation extraction + probing + figures).

---

## Repository Structure

```
.
├── README.md      
├── requirements.txt         
├── unlearning_probing_experiment  
├── .gitignore              
└── results/                   
    ├── fig1_probe_accuracy.png
    ├── fig2_pca.png
    ├── fig3_summary.png
    ├── training_curves.png
    ├── probe_results.csv
    └── results.json
```

---

## Citation

If you use this work or build on it, please cite the OPT-OUT paper that motivated this evaluation framework:

```bibtex
@inproceedings{choi2025opt,
  title={Opt-out: Investigating entity-level unlearning for large language models via optimal transport},
  author={Choi, Minseok and Rim, Daniel and Lee, Dohyun and Choo, Jaegul},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={28280--28297},
  year={2025}
}
```

And the TOFU benchmark:

```bibtex
@article{maini2024tofu,
  title={Tofu: A task of fictitious unlearning for llms},
  author={Maini, Pratyush and Feng, Zhili and Schwarzschild, Avi and Lipton, Zachary C and Kolter, J Zico},
  journal={arXiv preprint arXiv:2401.06121},
  year={2024}
}
```

