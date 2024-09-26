# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null values using .isnull() function.
3. Import LabelEncoder and encode the dataset.
4.Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5. Predict the values of arrays.
6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7. Predict the values of array.
8. Apply to new unknown values.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: NAGALAPURAM HASIF
RegisterNumber: 212223100036 
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
data = pd.read_csv("Salary_EX7.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
plt.figure(figsize=(20, 8))
plot_tree(dt, feature_names=x.columns, filled=True)
plt.show()
```
## Output:
## HEAD(),INFO(),null():

![image](https://github.com/user-attachments/assets/1ed69c7f-30b9-40c5-ac16-90cd68eb2822)


## Converting string literals to numerical values using label encoder:

![image](https://github.com/user-attachments/assets/d770a3a0-29d2-4470-b62a-feab3c9629b2)


## MEAN SQUARE ERROR:

![image](https://github.com/user-attachments/assets/082ca8f3-721a-46a2-80f9-6cef76854321)


## R2(Variance):

![image](https://github.com/user-attachments/assets/2bf54411-2495-4299-b324-17e4d5a2e7e0)


## DATA PREDICTION & DECISION TREE REGRESSOR FOR PREDICTING THE SALARY OF THE EMPLOYEE:

![image](https://github.com/user-attachments/assets/4488f9f9-5f02-4636-84b6-7080d93125b0)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
