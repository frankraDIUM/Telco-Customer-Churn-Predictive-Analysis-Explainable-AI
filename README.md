# Telco Customer Churn: Predictive Analysis & Explainable AI

## Project Overview

This project focuses on predicting customer attrition (churn) for a telecommunications company. By leveraging machine learning and **Explainable AI (SHAP)**, we identify high-risk customers and uncover the specific service and contractual factors driving their departure.

The final model achieves an **Area Under the Curve (AUC) of $0.838$**, providing a robust engine for proactive retention strategies.

---

##  Technical Workflow

### 1. Data Engineering

- **Dataset**: IBM Telco Customer Churn (7043 rows, 21 features)  
- **Cleaning**:
  - Filled 11 missing `TotalCharges` values (tenure = 0 customers) with 0
  - Converted binary Yes/No fields → 1/0
  - Collapsed "No internet service" → "No" for add-on services
  - Dropped `customerID`
- **Encoding**:
  - One-hot encoding for multi-category features (drop_first=True)
- **Feature Engineering**:
  - Created `AvgChargePerMonth` = MonthlyCharges / max(tenure, 1)
- **Imbalance handling**:
  - SMOTE applied only to training data
- **Split**: 80/20 train-test (random_state=42)

### 2. Modeling & Optimization

**Models trained**:
- Logistic Regression (with StandardScaler)
- Random Forest

**Hyperparameter tuning**:
- Logistic Regression → GridSearchCV → best `C=10`
- Random Forest → RandomizedSearchCV → best parameters:
  - `n_estimators=300`
  - `max_depth=20`
  - `min_samples_split=5`
  - `min_samples_leaf=1`
  - `class_weight='balanced_subsample'`

**Final test performance** (1409 samples):

| Model                        | Accuracy | ROC AUC | Precision (1) | Recall (1) | F1 (1) |
|------------------------------|----------|---------|---------------|------------|--------|
| Tuned Logistic Regression    | 77.0%    | 0.835   | 0.55          | **0.70**   | 0.62   |
| Tuned Random Forest          | **77.9%**| 0.834   | **0.58**      | 0.60       | 0.59   |

Both models significantly outperform the no-skill baseline (73.5% accuracy, 0.500 AUC).

### 3. Explainability (SHAP)

- **Tool used**: SHAP (TreeExplainer on tuned Random Forest)
- **Global importance** (mean |SHAP value|):
  1. `PaymentMethod_Electronic check` → +0.09
  2. `InternetService_Fiber optic` → +0.09
  3. `AvgChargePerMonth` → +0.06
  4. `tenure` → +0.05 (low tenure increases risk)
- **Beeswarm plot insights**:
  - Electronic check (when = 1) → strongly positive SHAP → large churn increase
  - Fiber optic (when = 1) → mostly positive SHAP → higher churn risk
  - Low tenure → positive SHAP (risk), high tenure → negative SHAP (protective)
  - Two-year contract & dependents → protective when present

## Business Insights (SHAP Analysis)

SHAP analysis clearly reveals the strongest churn drivers:

- **Electronic check payment method** — by far the largest average impact (+0.09 mean |SHAP|). When used, it consistently pushes churn probability upward.
- **Fiber optic internet service** — equally strong driver (+0.09). Customers on fiber optic show markedly higher churn risk compared to DSL or no internet.
- **High average monthly charge** — significant contributor (+0.06). Higher perceived cost correlates with increased likelihood to leave.
- **Low tenure** — early-stage customers (first 6–12 months) are highly vulnerable; long tenure becomes strongly protective.
- Protective factors (negative SHAP when present): two-year contracts, having dependents.

These patterns are consistent with telecom domain knowledge and provide high-confidence targets for intervention.

## Business Recommendations

**Top-priority actions based on SHAP importance:**

1. **Urgently migrate electronic check users to automatic payments**  
   (credit card / bank transfer) — offer incentives (discounts, waived fees), send targeted reminders.

2. **Deep-dive retention strategy for fiber optic customers**  
   Investigate satisfaction drivers (price, reliability, support quality) → introduce loyalty discounts, enhanced bundles, or proactive technical support outreach.

3. **Strengthen early-tenure engagement**  
   First 6–12 months are critical → implement onboarding programs, satisfaction check-ins, introductory offers, and personalized value communication.

4. **Address price sensitivity**  
   Monitor customers with high AvgChargePerMonth → offer tiered pricing, value-add services (e.g., security, streaming bundles), or targeted retention discounts.

5. **Amplify longer-contract incentives**  
   Two-year contracts are protective → deepen discounts, add perks (free upgrades, priority support) to encourage uptake.

Implementing these focused interventions has the highest potential to reduce churn and improve customer lifetime value.

## Technologies Used

- Python 3.8+
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- imbalanced-learn (SMOTE)
- shap
