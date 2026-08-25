import streamlit as st
import pandas as pd
import numpy as np
import joblib


# Load saved objects
scaler = joblib.load("scaler.pkl")
ohe = joblib.load("ohe.pkl")
model = joblib.load("logistic_model.pkl")


# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊"
)

st.title("📊 Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn.")


# Customer Information

tenure = st.number_input(
    "Tenure (Months)",
    min_value=0,
    value=12
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ]
)

internet_service = st.selectbox(
    "Internet Service",
    [
        "DSL",
        "Fiber optic",
        "No"
    ]
)

contract = st.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)

senior_citizen = st.selectbox(
    "Senior Citizen",
    [0, 1]
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No"]
)

paperless_billing = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)


# Prediction

if st.button("Predict Churn"):

    input_data = pd.DataFrame({
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
        "PaymentMethod": [payment_method],
        "InternetService": [internet_service],
        "Contract": [contract],
        "SeniorCitizen": [senior_citizen],
        "StreamingMovies": [streaming_movies],
        "PaperlessBilling": [paperless_billing]
    })


    numerical_features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "SeniorCitizen"
    ]

    categorical_features = [
        "PaymentMethod",
        "InternetService",
        "Contract",
        "StreamingMovies",
        "PaperlessBilling"
    ]


    # Scale numerical features
    numerical_data = scaler.transform(
        input_data[numerical_features]
    )


    # Encode categorical features
    categorical_data = ohe.transform(
        input_data[categorical_features]
    )


    # Combine features
    final_data = np.hstack([
        numerical_data,
        categorical_data
    ])


    # Prediction
    prediction = model.predict(final_data)


    if prediction[0] == "Yes":
        st.error("⚠️ Customer is likely to churn.")

    else:
        st.success("✅ Customer is unlikely to churn.")

        