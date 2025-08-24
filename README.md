# Malnutrition Prediction Pipeline

This repository contains a pipeline for predicting malnutrition classes using XGBoost models with Optuna hyperparameter tuning and SHAP feature explanation.

## Dataset Overview

* **Entries:** 462
* **Columns:** 173
* **Target Variables:**

  * `STUNTING_CLASS` (Normal, Moderate)
  * `WASTING_CLASS` (Normal)
  * `UNDERWEIGHT_CLASS` (Normal, Moderate)
* **Missingness:** Mostly minimal, except `ADMLEVEL` (\~98% missing).

## Target Distribution

### STUNTING\_CLASS

* Normal: 388
* Moderate: 74

### WASTING\_CLASS

* Normal: 462

### UNDERWEIGHT\_CLASS

* Normal: 416
* Moderate: 46

## Feature Categorization

* **Categorical:** `SEX_AGE_YEAR`, `WASTING_CLASS`, `UNDERWEIGHT_CLASS`, `SEX` (for STUNTING\_CLASS)
* **Numerical:** `AGE_START`, `HAZ_MEAN`, `WAZ_MEAN`, `WHZ_MEAN`, `AGE_END`, `YEAR_MID`

## Model Tuning & Evaluation

* **Method:** XGBoost with Optuna hyperparameter optimization
* **Evaluation Methods:** baseline, SMOTE, cost-sensitive
* **Performance Metrics:** Accuracy, Balanced Accuracy, Macro F1, Kappa

### STUNTING\_CLASS

* **Best Optuna parameters:**

```json
{
  "max_depth": 4,
  "learning_rate": 0.09399106736303652,
  "subsample": 0.9603818485758903,
  "colsample_bytree": 0.85303115542259,
  "reg_alpha": 0.9048933758318083,
  "reg_lambda": 4.8948207855748545e-08,
  "min_child_weight": 7,
  "n_estimators": 271
}
```

* **Top SHAP Features:** `HAZ_MEAN`, `WAZ_MEAN`, `YEAR_MID`, `WHZ_MEAN`, `AGE_END`

### UNDERWEIGHT\_CLASS

* **Best Optuna parameters:**

```json
{
  "max_depth": 4,
  "learning_rate": 0.04027374113128733,
  "subsample": 0.6111197488491856,
  "colsample_bytree": 0.9113879109186427,
  "reg_alpha": 0.012347799537506999,
  "reg_lambda": 0.32211369902182146,
  "min_child_weight": 1,
  "n_estimators": 397
}
```

* **Top SHAP Features:** `WAZ_MEAN`, `HAZ_MEAN`, `WHZ_MEAN`, `YEAR_MID`, `STUNTING_CLASS_Moderate`

### WASTING\_CLASS

* Only one class present (Normal), model not trained.

## Key Insights

* **STUNTING\_CLASS** is heavily influenced by `HAZ_MEAN` and `WAZ_MEAN`.
* **UNDERWEIGHT\_CLASS** is heavily influenced by `WAZ_MEAN` and `HAZ_MEAN`.
* Cost-sensitive methods slightly improve performance for imbalanced classes, though differences were not statistically significant according to paired t-tests.

## Usage

1. Preprocess your dataset with the same structure.
2. Run Optuna tuning for XGBoost to find optimal hyperparameters.
3. Evaluate using baseline, SMOTE, or cost-sensitive approaches.
4. Interpret model predictions using SHAP feature importance.

## Environment

* Python 3.10+
* Libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `optuna`, `shap`, `mlflow`

## License

This repository is for educational and research purposes.
