################################ lib
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
import os
import numpy as np

################################ data

"""
ID: Unique ID of the record
Customer_ID: Unique ID of the customer
Month: Month of the year
Name: The name of the person
Age: The age of the person
SSN: Social Security Number of the person
Occupation: The occupation of the person
Annual_Income: The Annual Income of the person
Monthly_Inhand_Salary: Monthly in-hand salary of the person
Num_Bank_Accounts: The number of bank accounts of the person
Num_Credit_Card: Number of credit cards the person is having
Interest_Rate: The interest rate on the credit card of the person
Num_of_Loan: The number of loans taken by the person from the bank
Type_of_Loan: The types of loans taken by the person from the bank
Delay_from_due_date: The average number of days delayed by the person from the date of payment
Num_of_Delayed_Payment: Number of payments delayed by the person
Changed_Credit_Card: The percentage change in the credit card limit of the person
Num_Credit_Inquiries: The number of credit card inquiries by the person
Credit_Mix: Classification of Credit Mix of the customer
Outstanding_Debt: The outstanding balance of the person
Credit_Utilization_Ratio: The credit utilization ratio of the credit card of the customer
Credit_History_Age: The age of the credit history of the person
Payment_of_Min_Amount: Yes if the person paid the minimum amount to be paid only, otherwise no.
Total_EMI_per_month: The total EMI per month of the person
Amount_invested_monthly: The monthly amount invested by the person
Payment_Behaviour: The payment behaviour of the person
Monthly_Balance: The monthly balance left in the account of the person
Credit_Score: The credit score of the person
"""

os.chdir('C:\\Users\\yurys\\Documents\\PyCharm\\SirotaProject')
data_file = 'C:\\Users\\yurys\\Documents\\PyCharm\\SirotaProject\\model_data_scoring.csv'
df = pd.read_csv(data_file)
df_xx = ["Age",  "Annual_Income","Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Outstanding_Debt" ]


df_yy = ["Credit_Score"]
df = df[ df_xx + df_yy ]
df_describe = df.describe()

################################ модели
model = LogisticRegression(random_state=123, max_iter=1000 ).fit( df[df_xx], df[df_yy].values.ravel() )

df["Credit_Score_predict_1"] = (model.predict_proba( df[df_xx] )[:,1]*100).astype(int)

pickle.dump(model, open('model.pickle', 'wb'))

model_load = pickle.load(open('model.pickle', 'rb'))


