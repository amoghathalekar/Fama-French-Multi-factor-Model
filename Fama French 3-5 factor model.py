# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 20:21:39 2020

@author: Amogh

            #####   Fama-French Three and Five Factor models   #####

"""


import pandas as pd
import yfinance as yf
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns


            ##########   Creating a Fama-French THREE-Factor Model   ##########

# Preparing the data:
df = pd.read_csv("F-F_Research_Data_Factors_daily.CSV", skiprows = 4, skipfooter = 1, parse_dates = [0], index_col = [0], engine='python')
msft = yf.download("MSFT", start = "2016-12-31", end = "2019-12-31")
msft = msft.loc[:, "Adj Close"]
msft = msft.pct_change().dropna()
msft = msft * 100
df["MSFT"] = msft
df = df.dropna()
df = df.rename(columns = {"Mkt-RF":"MktPrem"})
df["MSFTPrem"] = df.MSFT - df.RF

# Checking out the Correlation Matrix:
df.corr()

# Preparing Pair-wise regressions: (Not very meaningful in this case)
sns.pairplot(df, kind="reg")
plt.plot()

# Creating a Multiple Regression Model: (3 Factor)
model = ols("MSFTPrem ~ MktPrem + SMB + HML", data = df)
results = model.fit()
print(results.summary())

# COMMENTS: The R-squared of 73% is good. Looking at the slope coefficients and the fact that
#           the variables are significant, we can say that Microsoft behaves like a growth stock
#           in this case. But here, the Alpha is Not Significant.



 #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


             ##########   Creating a Fama-French FIVE-Factor Model   ##########
                      #     (continued from three factor model)      #

#Preparing the data: (Exporting a new .csv file)
df = pd.read_csv("F-F_Research_Data_5_Factors_2x3_daily.CSV", skiprows = 3, parse_dates = [0], index_col = [0])
df = df.rename(columns = {"Mkt-RF":"MktPrem"})
msft = yf.download("MSFT", start = "2016-12-31", end = "2019-12-31")
msft = msft.loc[:, "Adj Close"]
msft = msft.pct_change().dropna()
msft = msft * 100
df["MSFT"] = msft
df = df.dropna().copy()
df["MSFTPrem"] = df.MSFT - df.RF

# Checking out the Correlation Matrix:
df.corr()

# Preparing Pair-wise regressions: (Not very meaningful in this case)
sns.pairplot(df, kind="reg")
plt.plot()


# Creating a Multiple Regression Model: (5 Factor)
model = ols("MSFTPrem ~ MktPrem + SMB + HML + RMW + CMA", data = df)
results = model.fit()
print(results.summary())

# COMMENTS: The R-Squared has increased to 75%!
#           BUT the vairable Robust minus Weak (RMW) has a high p-value and hence we can say that it is Not Significant.
#           So we should delete the RMW factor. (see below)  

# Creating a Multiple Regression Model: (4 Factor)
model = ols("MSFTPrem ~ MktPrem + SMB + HML + CMA", data = df)
results = model.fit()
print(results.summary())

# COMMENTS: So now the R-Squared is the same. And all the variables are Statistically significant.
#           We can also conclude that Microsoft beavhes like a large growth stock and like an aggressively investing company.
#           It is a 4 factor model that explains 75% of the variation in the Microsoft returns.






