import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math


df = pd.read_csv("mpg.csv")
df = df.drop('name', axis=1)
df = df.replace('?', np.nan)
df = df.dropna()
X = df.drop('mpg', axis=1)
y = df['mpg']


# Init model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 42)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model = LinearRegression(fit_intercept=1)
model.fit(X_train.values,y_train.values) 
print("coef 0")
print(model.coef_[0])

for idx, col_name in enumerate(X_train.columns):
    print("koefficienter for {} er {}".format(col_name, model.coef_[idx]))

# forudsigelse
y_predict = model.predict(X_test.values)


model_mse = mean_squared_error(y_predict, y_test)
print("Hovedfejl : ", model_mse)
print("mpg : ", math.sqrt(model_mse))

y_pred = model.predict([[6,232,100,2634,13,71,1]])
print("Svar på test 1")
print(y_pred[0])

#8,390,190,3850,8.5,70,1
y_pred = model.predict([[8,390,190,3850,8.5,70,1]])
print("Svar på test 2")
print(y_pred[0])

#8,307,130,3504,12,70,1
y_pred = model.predict([[8,307,130,3504,12,70,1]])
print("Svar på test 3")
print(y_pred[0])

#8,304,193,4732,18.5,70,1
y_pred = model.predict([[8,304,193,4732,18.5,70,1]])
print("Svar på test 4")
print(y_pred[0])

#4,119,82,2720,19.4,82,1
y_pred = model.predict([[4,97,46,1835,20.5,70,2]])
print("Svar på test 5")
print(y_pred[0])

#4,98,66,1800,14.4,78,1
y_pred = model.predict([[8,350,165,3693,11.5,70,1]])
print("Svar på test 6")
print(y_pred[0])

#4,105,75,2230,14.5,78,1
y_pred = model.predict([[4,105,75,2230,14.5,78,1]])
print("Svar på test 7")
print(y_pred[0])

#4,90,48,1985,21.5,78,2
y_pred = model.predict([[4,90,48,1985,21.5,78,2]])
print("Svar på test 8")
print(y_pred[0])
