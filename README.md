# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to implement the simple linear regression model for predicting the marks scored.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Use the standard libraries in python.
2. Set variables for assigning dataset values.
3. Import LinearRegression from the sklearn.
4. Assign the points for representing the graph.
5. Predict the regression for marks by using the representation of graph.
6. Compare the graphs and hence we obtain the LinearRegression for the given datas.


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: PRETHIVEERAJAN P
RegisterNumber:  212221230079
*/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

dataset=pd.read_csv('/content/std_score.csv')

dataset.head()

dataset.tail()

x=dataset.iloc[:,:-1].values   #.iloc[:,start_col,end_col]
y=dataset.iloc[:,1].values
print(x)
print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

regressor=LinearRegression()
regressor.fit(x_train,y_train)

#for train values
y_pred=regressor.predict(x_test)
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='green')
plt.title("Hours Vs Score(Training set)")
plt.xlabel("Hours")
plt.ylabel("Score")
plt.show()

#for test values
y_pred=regressor.predict(x_test)
plt.scatter(x_test,y_test,color='blue')
plt.plot(x_test,regressor.predict(x_test),color='black')
plt.title("Hours Vs Score(Test set)")
plt.xlabel("Hours")
plt.ylabel("Score")
plt.show()
```

## Output:
![simple linear regression model for predicting the marks scored](output1.png)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
