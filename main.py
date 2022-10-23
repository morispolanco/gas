!pip install pandas
!pip install numpy
!pip install matplotlib
!pip install seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#Read the data
df = pd.read_csv('gasoline.csv')

#Check the data
df.head()
df.describe()

#Check for missing values
df.isnull().sum()

#Check for outliers
sns.boxplot(x=df['price'])
sns.boxplot(x=df['tax'])
sns.boxplot(x=df['income'])
sns.boxplot(x=df['highway'])
sns.boxplot(x=df['dollars'])

#Remove outliers
df = df[df['price']<=6]
df = df[df['tax']<=6]
df = df[df['income']<=6]
df = df[df['highway']<=6]
df = df[df['dollars']<=6]

#Check for outliers again
sns.boxplot(x=df['price'])
sns.boxplot(x=df['tax'])
sns.boxplot(x=df['income'])
sns.boxplot(x=df['highway'])
sns.boxplot(x=df['dollars'])

#Check for correlation
df.corr()

#Plot the correlation
sns.heatmap(df.corr(), annot=True)

#Split the data
X = df[['tax', 'income', 'highway', 'dollars']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Train the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Check the intercept
regressor.intercept_

#Check the coefficients
regressor.coef_

#Predict the test set results
y_pred = regressor.predict(X_test)

#Compare the actual output values for X_test with the predicted values
df1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1

#Check the accuracy of the model
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#Check the accuracy of the model
print('R2 score:', metrics.r2_score(y_test, y_pred))

#Predict the price of gasoline
tax = float(input('Enter the tax: '))
income = float(input('Enter the income: '))
highway = float(input('Enter the highway: '))
dollars = float(input('Enter the dollars: '))

price = regressor.intercept_ + regressor.coef_[0]*tax + regressor.coef_[1]*income + regressor.coef_[2]*highway + regressor.coef_[3]*dollars
print('The price of gasoline is: ', price)
