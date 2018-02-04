'''
This code is written by vishnu deo gupta as a part of learning ML. The main focous of this code was to
handle non-numerical or categorical data. This code uses Label Encoding.. There are other process to,
I will learn about them pretty soon.
'''
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from nltk.classify import ClassifierI
from random import shuffle

df = pd.read_csv("mushrooms.csv")
df = df.sample(frac=1)

#below lines are used to handle non-numerical or categorical data  by label encoding.... till that for loop 
labelencoder = LabelEncoder()

for col in df.columns:
    df[col] = labelencoder.fit_transform(df[col])
####

X = np.array(df.drop(['class'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['class'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
#model = MLPClassifier()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
