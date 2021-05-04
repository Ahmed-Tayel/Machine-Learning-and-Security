# -*- coding: utf-8 -*-

import random
import ast
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import *
from sklearn import preprocessing
import sklearn.metrics
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import json
import os
from joblib import dump, load

# Data Preprocessing ####################################################################################
print("Data Preprocessing start ...")
#####

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'dataset/dataset.json')
# JSON file 
f = open (filename, "r")
db = []
for x in f:
  db.append(json.loads(x))

print(db[0])

print("Data Preprocessing end ...")
#####


##Extract Text and Labels ####################################################################################
print("Extract Text and Labels start ...")
#####

def extractFeatDB(db):
  x_data = []
  y_data = []
  for i in range(len(db)):
    commandStr = ''
    commandListStr = db[i]['lista_asm']
    commandListArr = ast.literal_eval(commandListStr)
    for j in range(len(commandListArr)):
      firstWord = commandListArr[j].split(' ')[0]
      commandStr += firstWord
      commandStr += ' '
    x_data.append(commandStr)
    y_data.append(db[i]['semantic'])
  dataLabels  = {'command': x_data, 'label': y_data}
  dbFrame = pd.DataFrame(dataLabels)
  return dbFrame

def extractFeatSample(db):
  x_data = []
  y_data = []
  commandStr = ''
  commandListStr = db['lista_asm']
  commandListArr = ast.literal_eval(commandListStr)
  for j in range(len(commandListArr)):
    firstWord = commandListArr[j].split(' ')[0]
    commandStr += firstWord
    commandStr += ' '
  x_data.append(commandStr)
  y_data.append(db['semantic'])
  dataLabels  = {'command': x_data, 'label': y_data}
  dbFrame = pd.DataFrame(dataLabels)
  return dbFrame

dbFrame = extractFeatDB(db)

sampleData  = extractFeatSample(db[0])
sampleData

print("Extract Text and Labels end ...")
#####


##Text Vectorization ####################################################################################
print("Text Vectorization start ...")
#####

vectorizer = CountVectorizer(stop_words='english')
X_all = vectorizer.fit_transform(dbFrame.command)
y_all = dbFrame.label

print(X_all.shape)
print(y_all.shape)

X_normalized = preprocessing.normalize(X_all, norm='l1')

print(X_all[0])
print(X_normalized[0])

print("Text Vectorization end ...")
#####

# STEP 1: Normalization Effect on Results ####################################################################################
print("STEP 1: Normalization Effect on Results ...")
#####


## Split Data
print("STEP 1.1: Split Data ...")
#####

################ Splitting Normalized Dataset
X_train_norm, X_test_norm, y_train, y_test = train_test_split(X_normalized, y_all, 
          test_size=0.2, random_state=15)

id = random.randrange(0,X_train_norm.shape[0])
print('%d ' %(id))

print("Train: %d - Test: %d" %(X_train_norm.shape[0],X_test_norm.shape[0]))

################ Splitting Not Normalized Dataset
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, 
          test_size=0.2, random_state=15)

id = random.randrange(0,X_train.shape[0])
print('%d ' %(id))

print("Train: %d - Test: %d" %(X_train.shape[0],X_test.shape[0]))

## Create and Fit Model
print("STEP 1.2: Create and Fit Model ...")
#####

model = svm.SVC(C=1.0, kernel='poly', degree=2, gamma='scale')
print('SVM Model created')

############### Normalized
print("STEP 1.2.1: Normalized Model ...")
#####

#Fit Model
model.fit(X_train_norm, y_train)

#Evaluate Model
acc_norm = model.score(X_test_norm, y_test)

#Prediction
y_pred = model.predict(X_test_norm)

#Accuracy
print("Accuracy Normalized %.3f" %acc_norm)

#Percision and Recall
print(classification_report(y_test, y_pred, labels=None, digits=3))

############### Not Normalized
print("STEP 1.2.2: Not Normalized Model ...")
#####

#Fit Model
model.fit(X_train, y_train)

#Evaluate Model
acc = model.score(X_test, y_test)

#Prediction
y_pred = model.predict(X_test)

#Accuracy
print("Accuracy Not Normalized %.3f" %acc)

#Percision and Recall
print(classification_report(y_test, y_pred, labels=None, digits=3))

## Update Data Splitting

################ Splitting Normalized Dataset
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_all, 
          test_size=0.2, random_state=15)

id = random.randrange(0,X_train.shape[0])
print('%d ' %(id))
print("Train: %d - Test: %d" %(X_train.shape[0],X_test.shape[0]))

# STEP 2: SVM Poly Degree Tuning ####################################################################################
print("STEP 2: SVM Poly Degree Tuning ...")
#####

## Create and Fit Model
print("STEP 2.1: Create and Fit Model ...")
#####

XAxisPolyDegree = []
YAxisEncryptPrecision = []
YAxisAccuracy = []

for degree in range(1,15):
  model = svm.SVC(C=1.0, kernel='poly', degree=degree, gamma='scale')
  # Fit Model
  model.fit(X_train, y_train)
  #Prediction
  y_pred = model.predict(X_test)
  acc = model.score(X_test, y_test)
  report = classification_report(y_test, y_pred, labels=None, digits=3,output_dict=True)
  XAxisPolyDegree.append(degree)
  YAxisEncryptPrecision.append(report['encryption']['precision'])
  YAxisAccuracy.append(acc)

plt.figure(figsize = [10, 5])
plt.subplot(1,2,1)
plt.xlabel('Polynomial Degree')
plt.ylabel('Encryption Precision')
plt.plot(XAxisPolyDegree,YAxisEncryptPrecision)

plt.subplot(1,2,2)
plt.xlabel('Polynomial Degree')
plt.ylabel('Accuracy')
plt.plot(XAxisPolyDegree,YAxisAccuracy)

# STEP 3: SVM VS Gaussian Naive Bayes ####################################################################################
print("STEP 3: SVM VS Gaussian Naive Bayes ...")
#####

## Create and Fit Model
print("STEP 3.1: Create and Fit Model ...")
#####

#SVM
model = svm.SVC(C=1.0, kernel='poly', degree=3, gamma='scale')
# Fit Model
model.fit(X_train, y_train)
#Prediction
y_pred = model.predict(X_test)
acc = model.score(X_test, y_test)
report = classification_report(y_test, y_pred, labels=None, digits=3,output_dict=True)
encPrecision = report['encryption']['precision']
print("Encryption Precision for SVM is: %.3f" %encPrecision)
print("Accuracy for SVM is: %.3f" %acc)

#Gaussian Naive Bayes
model = GaussianNB()
# Fit Model
model.fit(X_train.toarray(), y_train)
#Prediction
y_pred = model.predict(X_test.toarray())
acc = model.score(X_test.toarray(), y_test)
report = classification_report(y_test, y_pred, labels=None, digits=3,output_dict=True)
encPrecision = report['encryption']['precision']
print("Encryption Precision for Gaussian Naive Bayes is: %.3f" %encPrecision)
print("Accuracy for Gaussian Naive Bayes is: %.3f" %acc)

# STEP 4: Noise Stress Test ####################################################################################
print("STEP 4: Noise Stress Test ...")
#####

## Insert Noise into Training set
print("STEP 4.1: Insert Noise into Training set ...")
#####

# randomly permute a percentage of training labels
def createNoiseData(y_all , percentage):
  y_all = np.copy(y_all)
  y_all_noisy = np.copy(y_all)
  ix_size = int(percentage * len(y_all_noisy))
  ix = np.random.choice(len(y_all_noisy), size=ix_size, replace=False) 
  b = y_all[ix]
  np.random.shuffle(b)
  y_all_noisy[ix] = b
  #y_all_noisy = pd.DataFrame(y_all_noisy, columns=['label'])
  return y_all_noisy

## Create and Fit Model

x_axisNoise = []

y_axisSVMPrecision = []
y_axisNBPrecision = []

y_axisSVMAcc = []
y_axisNBAcc = []

for n in range(5,50,5):
  n = n/100
  #Create Noisy Data
  y_train_noisy = createNoiseData(y_train.to_numpy() , n)
  x_axisNoise.append(n)
  
  #SVM
  model = svm.SVC(C=1.0, kernel='poly', degree=3, gamma='scale')
  # Fit Model
  model.fit(X_train, y_train_noisy)
  #Prediction
  y_pred = model.predict(X_test)
  acc = model.score(X_test,y_test)
  report = classification_report(y_test, y_pred, labels=None, digits=3,output_dict=True)
  encPrecision = report['encryption']['precision']
  y_axisSVMPrecision.append(encPrecision)
  y_axisSVMAcc.append(acc)

  #Gaussian Naive Bayes
  model = GaussianNB()
  # Fit Model
  model.fit(X_train.toarray(), y_train_noisy)
  #Prediction
  y_pred = model.predict(X_test.toarray())
  acc = model.score(X_test.toarray(),y_test)
  report = classification_report(y_test, y_pred, labels=None, digits=3,output_dict=True)
  encPrecision = report['encryption']['precision']
  y_axisNBPrecision.append(encPrecision)
  y_axisNBAcc.append(acc)

plt.figure(figsize = [10, 5])
plt.subplot(1,2,1)
plt.plot(x_axisNoise,y_axisSVMPrecision,label = 'SVM')
plt.plot(x_axisNoise,y_axisNBPrecision,label = 'GaussNB')
plt.xlabel('Noise Percentage')
plt.ylabel('Encryption Precision')
plt.legend()

plt.subplot(1,2,2)
plt.plot(x_axisNoise,y_axisSVMAcc,label = 'SVM')
plt.plot(x_axisNoise,y_axisNBAcc,label = 'GaussNB')
plt.xlabel('Noise Percentage')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# STEP 5: Update Final Model ####################################################################################
print("STEP 5: Update Final Model ...")
#####

#SVM
model = svm.SVC(C=1.0, kernel='poly', degree=3, gamma='scale')
# Fit Model
model.fit(X_train, y_train)

print("STEP 5.1: Save Final Model ...")
#####
dump(model, 'svmModel.joblib') 