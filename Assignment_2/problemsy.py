import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

import seaborn as sns

# Code source: Jaques Grobler
# License: BSD 3 clause

from scipy.spatial.distance import correlation

from statsmodels import datasets as moredatasets

from scipy.linalg import inv

import scipy.linalg as linalg

import networkx as nx

np.set_printoptions(precision=3)

## Aufgabe 1
df = pd.read_csv("./MathMarks.csv", header=0, sep=',')

df = df.drop(columns=['student_id'])
df

pd.plotting.scatter_matrix(df)

# same deal
sns.pairplot(df, )

# np.crrcoef: by default each row is a variable and each column an individual
# observation, but in the data frame, each row is an observatrion and the
# columns represent the variable, so use rowvar=False
cormat = np.corrcoef(df, rowvar=False)
cormatdf = pd.DataFrame(cormat, columns=list(df.columns), 
        index=list(df.columns))

sns.heatmap(cormatdf, center=0, vmin=-1, vmax=1,
        cmap='viridis')

D = inv(cormat)
D

S = np.array([D[i,i] for i in range(5)])
S = (S - 1)/S
S

### fitting
tname = df.columns[0]
y = df[[tname]] # predict variable
x = df.drop(columns=tname) # explain variables
x
regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(x, y)
r2_score(regr.predict(x),y)

r2scores = np.zeros(5)
j=0
for name in df.columns:
    y = df[[name]] # predict variable
    x = df.drop(columns=name) # explain variables
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    r2scores[j] = r2_score(regr.predict(x),y)
    j+=1

covmat = np.cov(df, rowvar=False)
p = inv(covmat)

scale = np.diagonal(p)
scale = np.sqrt(scale)

K = -p / scale
K = K.T
K = K / scale
K = K.T
K

ymech = df[['mechanics']]
yvect = df[['vectors']]
xtrain = df.drop(columns=["mechanics", "vectors"])
regr = linear_model.LinearRegression()
regr.fit(xtrain, ymech)
ymech_pred = regr.predict(xtrain)
regr.fit(xtrain, yvect)
yvect_pred = regr.predict(xtrain)
res_mech = ymech_pred - ymech
res_vect = yvect_pred - yvect

res_vect = res_vect.to_numpy().flatten()
res_mech = res_mech.to_numpy().flatten()
np.correlate(res_vect, res_mech)

np.corrcoef(res_vect, res_mech)
correlation(res_vect, res_mech) # correlation DISTANCE is 1 - correlation!


g = nx.from_numpy_array(K > 0.25)

mylabels = list(df.columns)
mylabels = list(zip(range(5), mylabels))
mylabels = dict(mylabels)
mylabels

nx.draw_spring(g, labels=mylabels, )


## Problem 2
# I am not sure what N(2*X+1,0.5) + epsilon means when X is itself a rv.

x = np.random.normal(0,1,1000)
eps  = np.random.normal(0,0.5,1000)
y = np.random.normal(2*x + 1, 0.5) + eps
z = np.random.normal(5*x + 1, 1) + eps

mydata = {"x" : x, "y" : y, "z" : z}

mydata = pd.DataFrame(mydata)
mydata

sns.pairplot(mydata, )


#regr = linear_model.LinearRegression()
#regr.fit(mydata[["x"]], mydata[["y","z"]])
#xhat = regr.predict(mydata[["y","z"]].to_numpy())
#regr.fit(mydata[["z"]], mydata[["x","y"]])
#xhat = regr.predict(mydata[["x","y"]].to_numpy())

# y vs z 
regr = linear_model.LinearRegression()
#regr.fit(x, y)
regr.fit(mydata[["z"]], mydata[["x"]])
#regr.fit(z, mydata[["x"]])
zhat = regr.predict(mydata[["x"]])
regr.fit(mydata[["y"]], mydata[["x"]])
yhat = regr.predict(mydata[["x"]])



# based on the covariance matrix

covmat = np.cov(mydata, rowvar=False)
p = inv(covmat)
scale = np.diagonal(p)
scale = np.sqrt(scale)
K = -p / scale
K = K.T
K = K / scale
K = K.T
K

regr = linear_model.LinearRegression()
regr.fit(X=mydata[["x"]], y=mydata[["z"]])
zhat = regr.predict(mydata[["x"]])
regr = linear_model.LinearRegression()
regr.fit(X=mydata[["x"]], y=mydata[["y"]])
yhat = regr.predict(mydata[["x"]])

zhat = zhat.flatten()
yhat = yhat.flatten()
np.corrcoef(z - zhat, y - yhat)
1 - correlation(y-yhat, z-zhat)

sns.heatmap(K, center=0, cmap="viridis")

sns.heatmap(np.corrcoef(mydata, rowvar=False), center=0, cmap="viridis")



### Aifgabe 3
mu = np.zeros(2)
sig = np.array([[1,0.6], [0.6,1]])

Z = np.random.multivariate_normal(mu, sig, 10)
np.corrcoef(Z, rowvar=False)
Z = np.random.multivariate_normal(mu, sig, 100)
np.corrcoef(Z, rowvar=False)
Z = np.random.multivariate_normal(mu, sig, 1000)
np.corrcoef(Z, rowvar=False)


