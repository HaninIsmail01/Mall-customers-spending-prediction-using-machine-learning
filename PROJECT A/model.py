import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import joblib

mall_customers = pd.read_csv(r'PROJECT A/dataset/Mall_Customers.csv')

mall_customers_reg= pd.get_dummies(mall_customers, columns=['Genre'])
mall_customers_reg.head()

X_reg = mall_customers_reg.drop(['Spending Score (1-100)'], axis =1)
Y_reg = mall_customers_reg['Spending Score (1-100)']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, Y_reg,test_size=0.3, random_state=42)

linear_regressor = LinearRegression()
linear_regressor.fit(X_train_reg, y_train_reg)
Y_pred_linear = linear_regressor.predict(X_test_reg)

joblib.dump(linear_regressor, r'PROJECT A/model.pkl')




