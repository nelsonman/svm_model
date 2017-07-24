#Importing pandas
import pandas as pd

import numpy as np

#Importing requests
import requests

#Importing JSON
import json

# Importing csv
import csv

# IMporting the date package 
import datetime
from datetime import timedelta, date
import time

# Importing sk-learn
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer



#Defining list
date_list=[]
usage_list = []

# DEfining a function that iterates over date in
# a for loop over range
def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

#Giving the start date and end date for the data file
start_date = date(2016, 1, 1)
end_date = date(2017, 1, 1)

# RUnning the for loop (with all the functions)
for x in daterange(start_date, end_date):

	#Setting Parameters (Start date and end date)
	parameters = {"startDate":x, "endDate":x}

	# Make a get requests with the parameters
	response = requests.get("http://api.smsbrain.in/1.2/appsms/dailyUsage.php",params = parameters)

	#Print the status code
	#print(response.status_code)

	# Loading the data in Json
	data = json.loads(response.text)	

	#Extending the content to a python list

	usage_list.extend([(data)])
	date_list.extend([time.mktime(datetime.datetime.strptime(str(x), "%Y-%m-%d").timetuple())])

	# Alloting a sleep time
	#time.sleep(1)
	
    


#print(date_list)


# Printing the header dictionary of content
#print(response.headers)
#print(response.headers["content-type"])

# creating a dataframe 
"""
sms_usage_df = pd.DataFrame({
	 'Date': np.array(date_list),
     'Usage': np.array(usage_list).ravel()
         })

usage_df = pd.DataFrame({
	'Usage': usage_list
	})

date_df = pd.DataFrame({
	'Date': date_list
	})
"""

# Printing dataframe
#print (sms_usage_df)

# Defining and creating a matrix

w, h = 2, 366;
Matrix = [[0 for x in range(w)] for y in range(h)] 

count = 0

for x in date_list:
	Matrix [count] [0] = x

	count = count + 1

count = 0

for x in usage_list:
	Matrix [count] [1] = x
	count = count + 1

#Printing a Matrix
print (Matrix)


#df = sms_usage_df
 
# Write list to csv :
#df.to_csv('sms.csv', index=True, header=True)

# SVM MODEL

X = Matrix
y = np.array(usage_list)



print ("\n The y array is as such:\n", y)

clf = svm.SVC(C=1.0, cache_size=200, coef0=0.0, degree=3, gamma='auto',
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

clf.fit(X, y)

#d = np.array(time.mktime(datetime.datetime.strptime(str("2016-01-03"), "%Y-%m-%d").timetuple()))

print("\nPrediction: \n", clf.predict([[1483209000.0, 0]]))

print(clf.score(X, y))




