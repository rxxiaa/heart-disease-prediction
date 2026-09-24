# Heart Disease Prediction: Econometrics vs. Machine Learning

A project that predicts heart disease from clinical data. It compares **logistic regression**, an interpretable econometric model with odds ratios and p-values, against **Random Forest** and **Gradient Boosting**.

The project answers two questions a clinician would ask:

1. **Does the patient have heart disease?** Binary classification, used for screening.
2. **How severe is it?** Multiclass classification (severity 0–4), used for triage.

*Course project for Advanced Econometrics (Bucharest University of Economic Studies).*

---

## Dataset

- **Source:** [Heart Disease UCI](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data) (Kaggle), originally from the UCI Machine Learning Repository
- **Size:** 920 patients, 16 columns, collected from four sites (Cleveland, Hungary, Switzerland, VA Long Beach)
- **Target:** `num`, renamed to `target`: 0 = healthy, 1–4 = increasing disease severity
- **Class balance:** 44.7% healthy, 55.3% with disease. Severe classes 3 and 4 make up under 15% of the data.

The original short column codes (`cp`, `trestbps`, `thalch`, …) were renamed to readable names (`chest_pain_type`, `resting_bp`, `max_heart_rate`, …).

---

## Methodology

```
Load & rename → Recode impossible zeros → EDA → Missing-value treatment → Feature engineering
      → Label encoding → Stratified 80/20 split → Standardisation
      → Part I: binary logit (statsmodels) + VIF + test evaluation + ML comparison + error analysis
      → Part II: multinomial logit + ML comparison
```

**Exploratory analysis** covers distributions by age, sex, age group, chest pain type and cholesterol; the relationship between maximum heart rate and age; and a correlation matrix.

**Impossible values:** `cholesterol = 0` (172 patients) and `resting_bp = 0` (1 patient) are recoded as missing.

**Missing values** are handled with three rules:

- Categorical variables with more than 30% missing (`major_vessels`, `thalassemia`, `st_slope`) get a separate `"Unknown"` category, because the missing test is itself informative.
- Categorical variables with few missing values are filled with the mode.
- Numeric variables are filled with the median, which is robust to outliers.

**Feature engineering** adds two clinically motivated variables:

- `colesterol_ridicat`: high cholesterol, with the clinical threshold at 240 mg/dl.
- `scor_risc`: a composite risk score (0–3) that counts high cholesterol, high blood pressure (≥ 140) and high fasting blood sugar.

**Econometric model:** a binary logit estimated with `statsmodels`, reporting coefficients, p-values, odds ratios, pseudo-R² and a likelihood-ratio test, with a VIF check for multicollinearity.

**Model comparison:** logistic regression, Random Forest and Gradient Boosting are evaluated on the same stratified test set with Accuracy, Precision, Recall, F1 and ROC AUC.

---

## Results

### Part I: Binary classification (healthy vs. disease)

**Logit model fit:** pseudo-R² = 0.379, and the model is globally significant (LLR p ≈ 10⁻⁷²).

**Significant risk factors** (standardised features, p < 0.05):

| Variable | Odds ratio | Meaning |
|---|---|---|
| `major_vessels` | 2.48 | vessels affected at fluoroscopy |
| `thalassemia` | 1.99 | impaired myocardial perfusion |
| `st_depression` | 1.89 | exercise-induced ischaemia |
| `sex` (male) | 1.85 | male patients are at higher risk |
| `exercise_angina` | 1.62 | exercise-induced angina |
| `age` | 1.57 | risk increases with age |
| `cholesterol` | 1.43 | higher cholesterol, higher risk |
| `max_heart_rate` | 0.76 | protective: a healthy heart reaches higher rates |
| `chest_pain_type` | 0.63 | asymptomatic patients (code 0) carry the highest risk |

**Test set performance (184 patients):**

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Gradient Boosting | 0.832 | 0.814 | 0.902 | 0.856 | 0.903 |
| Random Forest | 0.804 | 0.811 | 0.843 | 0.827 | 0.921 |
| Logistic regression (sklearn, balanced) | 0.799 | 0.835 | 0.794 | 0.814 | 0.896 |
| Logit (statsmodels) | 0.815 | 0.821 | 0.853 | 0.837 | 0.893 |

Gradient Boosting has the best F1 and Random Forest the best AUC. The gap to logistic regression is small, however: 4.2 percentage points in F1 and 2.5 in AUC. **Logistic regression is kept as the final model** because its coefficients and odds ratios can be explained to clinicians, which matters in a regulated medical setting.

Error analysis on the best predictive model shows it correctly identifies 90% of sick patients but only 74% of healthy ones. Its errors are mostly false alarms rather than missed patients, which is the preferable trade-off for screening.

### Part II: Multiclass classification (severity 0–4)

| Model | Accuracy | F1 macro | F1 weighted |
|---|---|---|---|
| Random Forest | 0.576 | 0.422 | 0.582 |
| Gradient Boosting | 0.582 | 0.379 | 0.564 |
| Multinomial logistic regression | 0.500 | 0.363 | 0.526 |

Predicting severity is much harder. Classes 3 and 4 are rare, and the models mostly confuse adjacent severity levels. The models can tell whether a patient is ill, but not how ill.

### Key insights

- **Data quality changes conclusions:** 172 patients (from the Switzerland and VA Long Beach sites) had cholesterol recorded as 0, which is really a missing value. Left untreated, these zeros produced a spurious negative cholesterol effect (OR = 0.59). After recoding them as missing, cholesterol becomes a positive, significant risk factor (OR = 1.43).
- Exercise-test variables (`st_depression`, `max_heart_rate`, `exercise_angina`) are among the strongest predictors, stronger than routine blood tests.
- **Asymptomatic** patients have the highest disease rate (79%). This reflects "silent", advanced disease discovered incidentally.
- From the 50–60 age group onward, patients with disease outnumber healthy ones.

---



## How to run

```bash
git clone https://github.com/rxxiaa/heart-disease-prediction.git
cd heart-disease-prediction
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook heart_disease_prediction.ipynb
```

---

## Tech stack

Python · pandas · NumPy · statsmodels · scikit-learn · matplotlib · seaborn · Jupyter

---

## Limitations and future work

- **Label encoding** imposes an artificial order on nominal variables. One-hot encoding with a reference category would give cleaner coefficients.
- `scor_risc` is built from other features (VIF ≈ 7.9), so it overlaps with its components.
- Imputation is done on the full dataset before the train/test split. A stricter pipeline would fit it on the training set only.
- The data is old and from a limited population, so the model would need validation on current, more diverse patient data.
- The severe classes are under-represented. Resampling (e.g. SMOTE) or an ordinal model could improve Part II.

---

