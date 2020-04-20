#Aristeidopoulou Niki 2937
#Nikolaos Vosios 1643
#1h seira askhsewn
#proetoimasia arxeiwn prin thn ektelesh toy algoritmoy

import pandas as pd

#file: spambase 
tempfile = open ("spambase.csv",'w')
for i in range(57):
    line = "x"+str(i)+","
    tempfile.write(line)
line = 'category'+'\n'
tempfile.write(line)
with open ('spambase.data') as file:
    firstfile = file.readlines()
    for line in firstfile:
        tempfile.write(line)
file.close()
tempfile.close()

#file: default of credit card clients 
read_file = pd.read_excel ('default of credit card clients.xls')[1:]
firstcolumn = read_file.columns[0]
read_file = read_file.drop([firstcolumn],axis=1)
read_file.to_csv ('default of credit card clients.csv', index = None, header=True)