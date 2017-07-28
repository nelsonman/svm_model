import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import datetime
from datetime import datetime,date,timedelta
import time

from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.svm import SVR


df = pd.read_csv('/Users/nelsondsouza/Downloads/Datasets/SMS_brain/2016.csv', sep=',', header = 0)


#print (daf)
#print (df.head())


year = lambda x: datetime.strptime(x, "%Y-%m-%d" ).year
df['year'] = df['Date'].map(year)


day_of_week = lambda x: datetime.strptime(x, "%Y-%m-%d" ).weekday()
df['day_of_week'] = df['Date'].map(day_of_week)

month = lambda x: datetime.strptime(x, "%Y-%m-%d" ).month
df['month'] = df['Date'].map(month)

day = lambda x: datetime.strptime(x, "%Y-%m-%d" ).day
df['day'] = df['Date'].map(day)

# please read docs on how week numbers are calculate
#week_number = lambda x: datetime.strptime(x, "%Y-%m-%d" ).strftime('%V')
#df['week_number'] = df['Date'].map(week_number)

df['usage'] = df['Usage']

#print(df.head())

df = df.drop('Date', 1)
df = df.drop('Usage',1)
#print(df.head())

array = df.values

X = array[:,0:5]
Y = array[:,4]

#print (array[:,4])

#print ("Array:", array)

clf = svm.SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='auto', kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

clf.fit(X, Y)



"""
User input date


i = raw_input("Please enter the date you want prediction for : (format yyyy-mm-dd)")

date = pd.DataFrame({
	"Date": [(i)]
	})

year = lambda x: datetime.strptime(x, "%Y-%m-%d" ).year
date['year'] = date['Date'].map(year)

month = lambda x: datetime.strptime(x, "%Y-%m-%d" ).month
date['month'] = date['Date'].map(month)

day = lambda x: datetime.strptime(x, "%Y-%m-%d" ).day
date['day'] = date['Date'].map(day)

date['Prediction'] = 0

date = date.drop('Date', 1)

"""

test = pd.read_csv('/Users/nelsondsouza/test.csv', sep=',', header = 0)
array2 = test.values
y_test = array2[:,4]
test = test.drop('usage', 1)
test['Prediction'] = 0
ray = test.values

#print("RAY:", ray)

X_test = ray[:,0:5]

print(X_test)
print(y_test)


predicted = clf.predict(X_test)
test['Prediction'] = predicted

print("Prediction:", [test])

print(clf.score(X_test, y_test))
