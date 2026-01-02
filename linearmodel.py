import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# loading the dataset
iris=load_iris()
X=iris.data
y=iris.target
# splitting the dataset into training and testing sets
X_train, X_test, y_trian, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# converting to pandas DataFrame for better visualization
df_train=pd.DataFrame(X_train, columns=iris.feature_names)
df_train['target'] = y_trian
df_test=pd.DataFrame(X_test, columns=iris.feature_names)
df_test['target'] = y_test
print("Training Set:",df_train.head(), sep="\n")
print("\n Testing set:",df_test.head(),sep="\n")
# Now its a time to create a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_trian)
# Making predictions
predictions = model.predict(X_test)
print("\n Predictions:", predictions)

# printing the model coefficients
print("\nModel Coefficients:", model.coef_)
print("\nModel Intercept:", model.intercept_)

# printing the accuracy of the model
accuracy = model.score(X_test, y_test)