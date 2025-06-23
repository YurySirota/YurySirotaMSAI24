import streamlit as st
import pickle
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression

st.title("Yury Sirota MSAI24 project. Streamlit, docker, scoring tools" )

st.header("Input borrower data")

age = st.slider(
    label="Age = ",
    min_value=14,
    max_value=60,
    value=37,
    step=1
)

annual_income = st.slider(
    label="Annual Income = ",
    min_value=7000,
    max_value=180000,
    value=86000,
    step=1000
)

number_bank_accounts= st.slider(
    label="Number of Bank Accounts = ",
    min_value=0,
    max_value=15,
    value=1,
    step=1
) 

number_credit_card = st.slider(
    label="Number of Credit Card = ",
    min_value=0,
    max_value=15,
    value=0,
    step=1
)

number_of_loan = st.slider(
    label="Number of Loans = ",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)

outstanding_debt = st.slider(
    label="Outstanding Debt = ",
    min_value=0,
    max_value=5000,
    value=2500,
    step=100
)

interest_rate = st.slider(
    label="Interest Rate = ",
    min_value=1,
    max_value=35,
    value=18,
    step=1
)

os.getcwd()

st.header("Run credit scoring")

button_run = st.button("Run credit scoring")
if button_run == True:
    model_load = pickle.load(open('C:\\Users\\yurys\\Documents\\PyCharm\\SirotaProject\\model.pickle', 'rb'))

    df_predict = pd.DataFrame(columns=[
        ["Age", "Annual_Income", "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
         "Outstanding_Debt"]])
    df_predict.loc[0] = [age, annual_income, number_bank_accounts, number_credit_card, interest_rate, number_of_loan,
                         outstanding_debt]

    st.write("Probability to be good borrower" ,(model_load.predict_proba( df_predict  )[:,1]*100)[0].astype(int) )



