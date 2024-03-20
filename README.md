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
Developed by: HARISH B
RegisterNumber: 212223040061
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
![image](https://github.com/RakshithaK11/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139336455/32d647e7-f70f-4fba-a8e6-8b16201c475c)
![image](https://github.com/RakshithaK11/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139336455/e2633c37-45a7-4b28-83f1-77ab15765e1e)
![image](https://github.com/RakshithaK11/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139336455/38361f24-94ec-42de-aa6e-14d67d3304ad)
![image](https://github.com/RakshithaK11/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139336455/89fee1bc-aabc-4868-b12e-7019d1f117da)
![image](https://github.com/RakshithaK11/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139336455/41b02477-baa5-487c-9c06-264d895097da)
![image](https://github.com/RakshithaK11/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139336455/637b03a9-3ad4-4390-8dc4-7ccd82119d9f)
![image](https://github.com/RakshithaK11/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139336455/42e96505-13fb-457d-9d81-d76d1486d5a9)
![image](https://github.com/RakshithaK11/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139336455/5e57cad1-6a4d-409b-a63d-a71657d5d2f2)
![image](https://github.com/RakshithaK11/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139336455/75502ec7-8b17-40dd-a575-db8c6d0e6e97)
![image](https://github.com/RakshithaK11/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139336455/9baf5590-6c48-4a0f-8162-111459350534)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
