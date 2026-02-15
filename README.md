# COVID-19 Fake News Detector
🌐 **Project page:** https://nikolailen.github.io/covid19-fake-news-detector/

This repository contains a deep-learning NLP pipeline for binary classification of COVID-19-related text as `fake` or `real`.

The implementation is in a notebook-first format:
- Main notebook: `covid19_fakenews_stella_training.ipynb`
- Dataset: `final_combined_dataset.csv`
- Report-style project write-up: `index.md`
- Visual assets: `visuals/`

## Project Scope

The goal is to detect misinformation in COVID-19 textual content using:
- Stella text embeddings (`dunzhang/stella_en_1.5B_v5`)
- A PyTorch feed-forward classifier (`SimpleNN`)
- Stratified cross-validation for weight-decay selection
- Final retraining + held-out test evaluation

## Current Pipeline (Notebook)

`covid19_fakenews_stella_training.ipynb` includes:
1. Environment setup and dependency installation (`%pip install -r requirements.txt`)
2. Optional `flash-attn` installation with runtime checks and fallback
3. Local dataset loading with schema checks and cleanup
4. Dataset hash verification with `final_combined_dataset.sha256`
5. Embedding generation (Stella)
6. Stratified K-fold CV (`train_model_cv`)
7. Retraining with best weight decay (`retrain_with_best_decay`)
8. Evaluation (`evaluate_model`) with:
   - accuracy, classification report, confusion matrix
   - ROC/AUC from probabilities (`y_score`)
9. Artifact saving:
   - model weights: `stella_model.pth`
   - reproducibility metadata: `stella_model_config.json`

## Reported Results

From `index.md`, the reported final run metrics are:
- Accuracy: `95.5175%`
- F1-score: `95%` (fake), `96%` (real)
- AUC: `0.955124`

See `index.md` for full methodology, discussion, and references.

## Repository Layout

```text
.
|- covid19_fakenews_stella_training.ipynb
|- requirements.txt
|- final_combined_dataset.csv
|- final_combined_dataset.sha256
|- index.md
|- visuals/
`- README.md
```

## Quick Start

1. Open `covid19_fakenews_stella_training.ipynb` in Jupyter/Colab.
2. Run cells from top to bottom.
3. Ensure the dataset file is present locally:
   - `final_combined_dataset.csv`
4. Install dependencies (already included in the notebook):

```python
%pip install -r requirements.txt
```

## Notes

- The dataset is expected to have columns: `Text`, `Label`.
- Labels are normalized to lowercase and empty text rows are removed.
- `flash-attn` is optional and only attempted when CUDA is available.
- The notebook is kept clean for version control (no saved outputs/execution counts).
