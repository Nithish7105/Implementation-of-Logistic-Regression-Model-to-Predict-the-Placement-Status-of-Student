# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import pandas for data manipulation and sklearn for machine learning operations.
2.Load data from a CSV file using pandas, then preprocess it by removing unnecessary columns and handling missing values if any.
3.Divide the preprocessed data into training and testing sets.
4.Train a machine learning model, such as logistic regression (lr), on the training data.
5.Calculate accuracy, generate confusion matrix, and produce a classification report to assess model performance.
6.Utilize the trained model to make predictions on new data points, ensuring it's fitted on training data before predicting on the test set. 
 
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: NITHISH KUMAR
RegisterNumber: 212223040134 
*/
```
~~~
import pandas as pd
data = pd.read_csv("/content/Placement_Data.csv")
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = (y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
~~~

## Output:
![image](https://github.com/user-attachments/assets/5a915b37-b7b0-4122-a516-0731f4e09251)
![image](https://github.com/user-attachments/assets/97e7b035-8e97-441e-a959-433f0e4df888)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
