import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

dataset = pd.read_csv("data.csv")

dataset = dataset.drop(['Salary', 'Salary Currency'], axis=1)

X = dataset.drop('Salary in USD', axis=1)
Y = dataset['Salary in USD']

categorical_features = ['Job Title', 'Employment Type', 'Experience Level', 'Expertise Level', 'Company Location', 'Employee Residence', 'Company Size', 'Year']

from sklearn.preprocessing import LabelEncoder
label_encoders = {}
for feature in categorical_features:
    label_encoders[feature] = LabelEncoder()
    X[feature] = label_encoders[feature].fit_transform(X[feature])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

x_sc = StandardScaler()
X_train = x_sc.fit_transform(X_train)
X_test = x_sc.transform(X_test)

y_sc = StandardScaler()

Y_train = np.array(Y_train).reshape(-1, 1)
Y_train = y_sc.fit_transform(Y_train)

Y_test = np.array(Y_test).reshape(-1, 1)
Y_test = y_sc.transform(Y_test)

# Training SVR Model
from sklearn.svm import SVR

svr = SVR(kernel="rbf")
svr.fit(X_train, Y_train)

# Predicting the test set results
Y_pred = svr.predict(X_test)

# Inverse transforming the scaled predictions and actual values for meaningful comparison
Y_pred = y_sc.inverse_transform(Y_pred.reshape(-1, 1))
Y_test = y_sc.inverse_transform(Y_test)

# Calculating R-squared
from sklearn.metrics import r2_score
r_squared = r2_score(Y_test, Y_pred)

print(f"R-squared: {r_squared:.2f}")