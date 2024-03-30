from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.cluster import KMeans

#reading countries list
data = pd.read_csv('iris.csv')

#selecting data to plot
x = data.iloc[:, 1:4]

y = data.iloc[:, 4:5]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

print (X_test)
#gaussian naive basyan
gnb = GaussianNB()
model = gnb.fit(X_train.values, y_train.values.ravel())
y_pred = model.predict(X_test.values)
print(f"GaussianNB: Number of mislabeled points out of a total { X_test.shape[0] } points : {(y_test.values.ravel() != y_pred).sum()}" )

print(model.predict(np.array([[3.2,5.9,2.3]])))

#decussion tree
from sklearn import tree
clf = tree.DecisionTreeClassifier()
model = clf.fit(X_train.values, y_train.values.ravel())
y_pred = model.predict(X_test.values)
print(f"DecisionTreeClassifier: Number of mislabeled points out of a total { X_test.shape[0] } points : {(y_test.values.ravel() != y_pred).sum()}" )
print(model.predict(np.array([[3.2,5.9,2.3]])))