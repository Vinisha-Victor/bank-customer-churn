# Bank Customer Churn Prediction

Predicts whether a bank customer is likely to churn (leave the bank) based on their profile and account behaviour, with per-prediction explainability powered by SHAP.

**SkillInfyTech Data Science Internship — Project 1**

---

## Overview

Banks lose significant revenue when customers close their accounts. This project builds a machine learning pipeline that identifies customers at risk of churning *before* they leave, and explains **why** each prediction was made — turning a black-box model into something a bank's retention team could actually act on.

**Dataset:** [Bank Customer Churn dataset](https://www.kaggle.com/datasets/shubhammeshram579/bank-customer-churn-prediction) (10,000 customers, Kaggle)

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Class imbalance | SMOTE (imbalanced-learn) |
| Modelling | scikit-learn (Logistic Regression, Decision Tree), XGBoost |
| Explainability | SHAP |
| Backend API | FastAPI |
| Frontend | Gradio (Hugging Face Spaces) |
| Deployment | Render (API) + Hugging Face Spaces (UI) |

---

## Exploratory Data Analysis — Key Findings

- **Overall churn rate:** 20.4% of customers churned
- **Geography matters:** Germany has a churn rate of ~32%, roughly double France (~16%) and Spain (~17%)
- **Product count is a red flag:** Customers with 3-4 products churn at 83-100%, far higher than those with 1-2 products
- **Engagement predicts retention:** Inactive members churn at ~27% vs ~14% for active members
- **Gender gap:** Female customers churn more than male customers (25% vs 16%)
- **Age is the single strongest individual signal:** churn risk rises steadily with customer age

---

## Model Comparison

| Model | Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 73.0% | 0.471 | 0.743 |
| Decision Tree | 79.1% | 0.562 | 0.823 |
| **XGBoost** | **80.8%** | **0.581** | **0.846** |

XGBoost was selected as the production model based on its superior ROC-AUC and F1 score, both critical given the class imbalance in churn data.

---

## Explainability (SHAP)

Rather than treating the model as a black box, SHAP (SHapley Additive exPlanations) is used to explain:

- **Global feature importance** — which features drive churn predictions overall
- **Per-prediction explanations** — for any individual customer, which specific factors pushed their churn probability up or down

**Top global churn drivers (by mean SHAP value):**
1. Age
2. Number of Products
3. Active Membership status
4. Gender
5. Account Balance

This makes the model's decisions transparent and actionable — a retention team can see *exactly* why a customer was flagged as high-risk, not just that they were.

---

## Project Structure

```
bank-customer-churn/
├── data/                     # Dataset (gitignored)
├── notebooks/
│   └── churn_analysis.ipynb  # Full EDA → preprocessing → modelling → SHAP pipeline
├── scripts/
│   └── train_model.py        # Standalone training script (used at deploy time)
├── models/                   # Saved model artifacts (gitignored, generated at runtime)
├── app/
│   ├── main.py                # FastAPI prediction API
│   └── schemas.py             # Request/response schemas
├── requirements.txt
└── README.md
```

---

## Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/bank-customer-churn.git
cd bank-customer-churn

# Set up environment
python -m venv .venv
.venv\Scripts\Activate.ps1      # Windows
pip install -r requirements.txt

# Train the model
python scripts/train_model.py

# Run the API
uvicorn app.main:app --reload
```

---

## Live Demo

- **API (Render):** _coming soon_
- **Interactive UI (Hugging Face Spaces):** _coming soon_

---

## What Makes This Project Different

Most beginner churn-prediction projects stop at Logistic Regression and a confusion matrix. This one goes further:

- **SMOTE** to properly handle class imbalance instead of ignoring it
- **XGBoost** benchmarked against simpler baselines, not used blindly
- **SHAP explainability** at both the global and individual-prediction level
- **Production-style structure** — a training script separate from the notebook, a FastAPI backend, and a deployed frontend, mirroring a real ML engineering workflow rather than a one-off notebook