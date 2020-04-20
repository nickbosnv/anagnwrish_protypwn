#Aristeidopoulou Niki 2937
#Nikolaos Vosios 1643
#1h seira askhsewn
#Algorithmos kai 4 methodoi

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from pandas import DataFrame, read_csv
from numpy import genfromtxt

a = 0.001 #poso metavolis twn parametrwn
e = math.e
lenoffolds = 10 #posa fold xreiazomai
M = 10 #o akeraios ari8mos
TP = 0 #True Positive
TN = 0 #True Negative
FP = 0 #False Positive
FN = 0 #False Negative
P = 0 #Positive
N = 0 #Negative
results = [] #Apotelesmata methodwn

#arxikopoihsh twn meswn μj kai σj
def initialize_mj_sj(data_frame):
    inmj = []
    insj = []
    category = np.where(data_frame[:,-1] == 0)
    cat0 = data_frame[category]
    inmj.append(np.mean(cat0,axis = 0))
    insj.append(np.var(cat0[:,:-1]))
    category = np.where(data_frame[:,-1] == 1)
    cat1 = data_frame[category]
    inmj.append(np.mean(cat1,axis = 0))
    insj.append(np.var(cat1[:,:-1]))
    return (inmj,insj)

#Proetoimasia arxeiwn
dataframe = pd.read_csv('spambase.csv')
dataframe = dataframe.to_numpy(dtype =o bject)
mat_sort = np.argsort(dataframe[:,-1])
df = dataframe[mat_sort]

#Dhmioyrgia 10 folds me tyxaia epilogh
#diaxorismos ka8e kathgorias analogika se ka8e fold
i0=0
category = np.where(df[:,-1] == 0)
cat0 = df[category]
category = np.where(df[:,-1] == 1)
cat1 = df[category]
j0 = int(len(cat0)/lenoffolds)
i1 = len(cat0)
j1 = int(len(cat0)+(len(cat1)/lenoffolds))
folds = []
for i in range(9):
    df0 = df[i0:j0]
    df1 = df[i1:j1]
    dfmer = np.concatenate((df0,df1), axis = 0)
    folds.append(dfmer)
    i0 += int(len(cat0)/lenoffolds)
    j0 += int(len(cat0)/lenoffolds)
    i1 += int(len(cat1)/lenoffolds)
    j1 += int(len(cat1)/lenoffolds)
df0 = df[i0:len(cat0)]
df1 = df[i1:]
dfmer = np.concatenate((df0,df1), axis=0)
folds.append(dfmer)

#Arxikopoihsh timwn gia tis methodous
import sklearn.metrics as metrics

x_test = np.concatenate((df[i0:len(cat0),:-1],df[i1:,:-1]), axis=0)
y_test = np.concatenate((df[i0:len(cat0),-1],df[i1:,-1]), axis=0)
x_train = np.concatenate((df[0:(j0*9),:-1],df[i1:(j1*9),:-1]), axis=0)
y_train = np.concatenate((df[0:(j0*9),-1],df[i1:(j1*9),-1]), axis=0)
y_train = y_train.astype('int')
y_test = y_test.astype('int')

#Nearest Neighbor k-NN
from sklearn.neighbors import KNeighborsClassifier

k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)
name = 'Nearest Neighbor k-NN'
Acc = metrics.accuracy_score(y_test,y_pred)
F1 = metrics.f1_score(y_test,y_pred,average='weighted')
comment = 'k neighbors = '+str(k)
results.append({"Method":name,"Accuracy":Acc,"F1 score":F1,"Comments":comment})
print (name)
print ("Αccuracy:", Acc)
print ("F1 score: ", F1)

#Neural Networks
from sklearn.neural_network import MLPClassifier

hidden = 2 #krimmeno epipedo
k = 5 #neurwnes
solv = 'adam' #enallaktika sximata gia thn Stochastic Gradient Descent 'adam' kai gia thn Gradient Descent 'sgd'
nn = MLPClassifier(activation = 'logistic',solver = solv, hidden_layer_sizes = (k, hidden))
nn.fit(x_train,y_train)
y_pred = nn.predict(x_test)
name = 'Neural Networks'
Acc = metrics.accuracy_score(y_test,y_pred)
F1 = metrics.f1_score(y_test,y_pred,average='weighted')
comment = 'hidden = '+str(hidden)+' k = '+str(k)+' solver = '+solv
results.append({"Method":name,"Accuracy":Acc,"F1 score":F1,"Comments":comment})
print (name)
print ("Αccuracy:", Acc)
print ("F1 score: ", F1)

#Support Vector Machines
from sklearn import svm

tyker = 'linear' #Gia thn grammiki synartisi pyrhna 'linear' kai gia thn Gaussian synarthsh pryina 'rbf'
svm_clf = svm.SVC(kernel = tyker)
svm_clf.fit(x_train,y_train)
y_pred = svm_clf.predict(x_test)
name = 'Support Vector Machines'
Acc = metrics.accuracy_score(y_test,y_pred)
F1 = metrics.f1_score(y_test,y_pred,average='weighted')
comment = 'kernel = '+tyker
results.append({"Method":name,"Accuracy":Acc,"F1 score":F1,"Comments":comment})
print(name)
print ("Αccuracy:", Acc)
print ("F1 score: ", F1)

#Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(x_train,y_train)
y_pred = gnb.predict(x_test)
name = 'Naive Bayes classifier'
Acc = metrics.accuracy_score(y_test,y_pred)
F1 = metrics.f1_score(y_test,y_pred,average='weighted')
comment = None
results.append({"Method":name,"Accuracy":Acc,"F1 score":F1,"Comments":comment})
print (name)
print ("Αccuracy:", Acc)
print ("F1 score: ", F1)

#algoritmos san ton LVQ
for test in range(M): #M einai o akaireos ari8mos
    mj, sj = initialize_mj_sj(dataframe) #arxikopoihsh twn mj kai sj
    for i,data in enumerate(folds): #psaxnw ola ta fold
        if i == test: #krataw to test
            testdata=data #testing set
        else: #kanw ekpaideysh
            for row in range(len(data)): #training set
                x = data[row,:-1] #ta xaraktiristika x
                y = data[row,-1] #h pragmatikh kathgoria
                gaus = []
                for cat,m in enumerate(mj):
                    summ = np.linalg.norm(x-m[:-1]) #ypologizei to |x-mj|                
                    arith = math.pow((summ),2) #ypologismos toy arithmiti gia thn gausiani perioxi
                    paron = (math.pow((sj[cat]),2))*2 #ypologismos tou paranomasth gia thn gausiani perioxi
                    tempgaus = math.pow(e,(-(arith)/(paron)))
                    gaus.append(tempgaus)
                catj = gaus.index(max(gaus)) #thesi nikitrias perioxis
                if y == catj: #an to y einai iso me to Cj
                    mj[catj][:-1] = (1-a)*mj[catj][:-1] + a * x
                    dist = np.linalg.norm(x-mj[catj][:-1]) #ypoligzei to dist(x,mj) ths kathgorias Cj
                    sj[catj] = sj[catj] + a * dist                    
                else: #an den einai iso
                    mj[catj][:-1] = (1-a)*mj[catj][:-1] - a * x
                    dist = np.linalg.norm(x-mj[catj][:-1])
                    sj[catj] = sj[catj] - a * dist
                    x = data[row]
                    mj.append(x)
                    sinit = (np.mean(sj))*0.1
                    sj.append((np.mean(sj))*0.1)
                    
    for row in range(len(testdata)): #test set
        x = testdata[row,:-1] 
        y = testdata[row,-1] 
        gaus = []
        for cat,m in enumerate(mj):
            summ = np.linalg.norm(x-m[:-1])                   
            arith = math.pow((summ),2)
            paron = (math.pow((sj[cat]),2))*2
            tempgaus = math.pow(e,(-(arith)/(paron)))
            gaus.append(tempgaus)
        catj = gaus.index(max(gaus))
        if y == catj: #an epityxei tote exoyme True
            if catj == 1: #1 shmainei oti einai Negative
                N += 1
                TN += 1
            else: #alliws 0 einai Positive
                TP += 1
                P += 1
        else: #an apotyxei tote exoume False
            if catj == 1:
                N += 1
                FN += 1
            else:
                FP += 1
                P += 1              

name = 'Algoritmos'
Acc = (TP + TN) / (P + N)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2* ((Precision*Recall)/(Precision+Recall))
comment = 'M = '+str(M)
results.append({"Method":name,"Accuracy":Acc,"F1 score":F1,"Comments":comment})
print (name)
print ("Accuracy : ",Acc)
print ("F1 score: ", F1)

#Telika apotelesmata
sorted_results = sorted(results, key=lambda i: i['F1 score'])
print("---The results of the method in increasing order by F1 score---")
for m in sorted_results:
    print(m)
best_method = sorted_results[-1]
print('-------\\\\////-------')
print('Best method is :',best_method['Method'],'with Accuracy:',best_method['Accuracy'],end = '')
print(', F1 score :',best_method['F1 score'],'and variables:',best_method['Comments'])




