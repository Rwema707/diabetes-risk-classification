# Predicting Diabetes from Health Survey Data: Traditional ML vs Deep Learning

This project predicts whether a person has diabetes or prediabetes from their answers to the CDC's 2015
Behavioral Risk Factor Surveillance System (BRFSS) survey. It compares traditional machine-learning models
(scikit-learn) with neural networks (TensorFlow/Keras) on the same data, splits and metrics.

- **Report:** _link to PDF, if you add it_
- **Google Drive (raw data + saved models):(https://drive.google.com/drive/folders/1EnzGGV897jgvD8vfGWnc0qRj6ig7-13S?usp=sharing)
- **Video presentation:** _add link_

## Dataset

**CDC Diabetes Health Indicators**, UCI Machine Learning Repository (ID 891), derived from BRFSS 2015.

- 253,680 survey responses, 21 features (health conditions, lifestyle, demographics, access to care)
- Target `Diabetes_binary`: 0 = no diabetes, 1 = prediabetes or diabetes (about 14% of rows)
- Source: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators

The raw data is **not** committed to this repository. To run the notebook, either:

1. download `diabetes_binary_health_indicators_BRFSS2015.csv` (from the Drive folder above or from UCI) and put it in
   `data/`, or
2. do nothing: the notebook falls back to downloading the data with `ucimlrepo`.

## Repository structure

```
diabetes_classification.ipynb   main notebook (runs top to bottom)
requirements.txt                dependencies
figures/                        every plot used in the report
results/                        experiment log (experiments.csv / .md), test results, error analysis tables
logs/                           TensorBoard logs for every deep-learning run
data/                           (not committed) put the raw CSV here
models/                         (not committed) saved models, also in the Drive folder
```

## How to run

```bash
git clone <this repo>
cd <repo>
pip install -r requirements.txt
jupyter notebook diabetes_classification.ipynb      # then Run All
tensorboard --logdir logs                           # to inspect the deep-learning runs
```

On Google Colab: upload the notebook, uncomment the `!pip install` line in the first code cell, upload the CSV into a
`data/` folder, and choose Runtime > Run all. A full run takes roughly 15-25 minutes on a CPU.

All random seeds are fixed (`SEED = 42`), so re-running gives the same splits and very close scores.
Tested with Python 3.13, scikit-learn 1.9, TensorFlow 2.21, pandas 3.0.

## Method in brief

- **Cleaning:** removed exact duplicate rows, capped BMI at the 99.5th percentile of the training set.
- **Feature engineering:** BMI bands, cardiovascular risk count, healthy-habits score, poor-health days,
  age x BMI interaction, lack of access to care.
- **Split:** stratified 70 / 15 / 15 train / validation / test. Models are compared on validation; the test set is
  used once for the final two models.
- **Traditional ML (8 experiments):** logistic regression, random forest, histogram gradient boosting, randomised
  hyperparameter search, mutual-information feature selection, decision-threshold tuning.
- **Deep learning (8 experiments):** Sequential MLPs and a Functional-API wide-and-deep network with embeddings,
  fed through `tf.data`; experiments on capacity, dropout/L2/batch norm, learning rate, class weights, batch size and
  threshold.
- **Evaluation:** accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrices, ROC and PR curves, learning
  curves, permutation importance and an error analysis by age and BMI group.

## Results

Test-set results (34,422 rows the models never saw during training or selection):

| Model | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Gradient boosting (ML-6, tuned) | 0.50 | 0.584 | 0.167 | 0.260 | 0.817 | 0.438 |
| Gradient boosting (ML-6, tuned) | 0.233 | 0.381 | 0.624 | 0.473 | 0.817 | 0.438 |
| Wide-and-deep network (DL-6) | 0.50 | 0.580 | 0.162 | 0.254 | 0.817 | 0.439 |
| Wide-and-deep network (DL-6) | 0.257 | 0.386 | 0.606 | 0.472 | 0.817 | 0.439 |

The two families perform almost identically. Gradient boosting trains in seconds rather than minutes and is easier to
explain. Class weighting and threshold tuning change recall far more than the choice of algorithm does.

The full experiment log is in [`results/experiments.md`](results/experiments.md).

## Limitations

The data is self-reported, comes from a single year, and merges prediabetes with diabetes. Survey respondents who
do not know they have diabetes are labelled 0, so some "false positives" may be undiagnosed cases.
