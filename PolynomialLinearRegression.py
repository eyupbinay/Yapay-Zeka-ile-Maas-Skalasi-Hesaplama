#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("polynomial.csv",sep = ";")


polynomial_regression = PolynomialFeatures(degree = 4)

x_polynomial = polynomial_regression.fit_transform(df[['deneyim']])


reg = LinearRegression()
reg.fit(x_polynomial,df['maas'])


y_head = reg.predict(x_polynomial)
plt.plot(df['deneyim'],y_head,color= "red",label = "polynomial regression")
plt.legend()

plt.scatter(df['deneyim'],df['maas'])   
plt.show()


x_polynomial1 = polynomial_regression.fit_transform([[4.5]])
reg.predict(x_polynomial1)

