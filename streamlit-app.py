import streamlit as st
import joblib
import pandas as pd
import sklearn
import sys
import os

model = joblib.load("rf_model.pkl")

st.title("Классификатор одобрения кредита")

st.markdown("Введите данные клиента для оценки кредитной заявки")

age = st.number_input("Возраст", min_value=18, max_value=100, value=30)

occupation_status = st.selectbox(
    "Статус занятости",
    ["Employed", "Self-employed", "Unemployed", "Student", "Retired"]
)

years_employed = st.number_input(
    "Стаж работы (лет)", min_value=0.0, max_value=50.0, value=5.0
)

annual_income = st.number_input(
    "Годовой доход", min_value=0, value=50000
)

credit_score = st.number_input(
    "Кредитный рейтинг", min_value=300, max_value=850, value=650
)

credit_history_years = st.number_input(
    "Длительность кредитной истории (лет)", min_value=0.0, max_value=50.0, value=5.0
)

savings_assets = st.number_input(
    "Сбережения", min_value=0, value=10000
)

current_debt = st.number_input(
    "Текущая задолженность", min_value=0, value=5000
)

defaults_on_file = st.selectbox(
    "Были ли дефолты?",
    [0, 1]
)

delinquencies_last_2yrs = st.number_input(
    "Просрочки за последние 2 года", min_value=0, value=0
)

derogatory_marks = st.number_input(
    "Негативные отметки в кредитной истории", min_value=0, value=0
)

product_type = st.selectbox(
    "Тип кредитного продукта",
    ["Personal", "Mortgage", "Auto", "Education"]
)

loan_intent = st.selectbox(
    "Цель кредита",
    ["Education", "Business", "Medical", "Home", "Personal"]
)

loan_amount = st.number_input(
    "Сумма кредита", min_value=0, value=20000
)

interest_rate = st.number_input(
    "Процентная ставка (%)", min_value=0.0, max_value=50.0, value=10.0
)

debt_to_income_ratio = st.number_input(
    "Debt-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.3
)

loan_to_income_ratio = st.number_input(
    "Loan-to-Income Ratio", min_value=0.0, max_value=5.0, value=0.5
)

payment_to_income_ratio = st.number_input(
    "Payment-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.2
)

if st.button("Предсказать"):
    input_data = pd.DataFrame([{
        "age": age,
        "occupation_status": occupation_status,
        "years_employed": years_employed,
        "annual_income": annual_income,
        "credit_score": credit_score,
        "credit_history_years": credit_history_years,
        "savings_assets": savings_assets,
        "current_debt": current_debt,
        "defaults_on_file": defaults_on_file,
        "delinquencies_last_2yrs": delinquencies_last_2yrs,
        "derogatory_marks": derogatory_marks,
        "product_type": product_type,
        "loan_intent": loan_intent,
        "loan_amount": loan_amount,
        "interest_rate": interest_rate,
        "debt_to_income_ratio": debt_to_income_ratio,
        "loan_to_income_ratio": loan_to_income_ratio,
        "payment_to_income_ratio": payment_to_income_ratio
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"Кредит ОДОБРЕН\n\nВероятность одобрения: {probability:.2%}")
    else:
        st.error(f"Кредит ОТКЛОНЁН\n\nВероятность одобрения: {probability:.2%}")
