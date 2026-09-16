# Loan Default Risk Prediction

A machine learning project that predicts whether a borrower is likely to
default on a loan within the next two years using the
[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) dataset.

The project covers the complete machine learning pipeline:

- Data cleaning
- Train/test splitting
- Leakage-safe preprocessing
- Feature transformation
- Model training and comparison
- Model evaluation
- Streamlit deployment

---

##  Project Goal

Predict `SeriousDlqin2yrs`:

- `1` → Serious delinquency within two years
- `0` → No serious delinquency within two years

The project demonstrates an end-to-end credit-risk classification workflow.

---

## 📂 Project Structure

```text
├── loan_default_split_first.ipynb
├── app.py
├── model.pkl
├── scaler.pkl
├── feature_names.pkl
└── README.md
