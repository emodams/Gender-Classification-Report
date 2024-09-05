# Gender-Classification-Report

# REPORT OVERVIEW
The objective of the "Gender Classification Report" is to develop and evaluate a robust model for gender classification using a diverse and comprehensive dataset. 
it provides a detailed analysis of the model's performance on the test dataset.

This model aims to achieve high accuracy while minimizing biases, ensuring fair and ethical outcomes. The report will provide insights into the model's performance, highlight areas for improvement, and offer recommendations for its application in real-world scenarios.initiatives.
       
- Target Variable : Gender(Classification of Male and Female)



## Model Performance Metrics:
- ### Precision:

- Male: 0.98%
- Female: 0.97%
Precision indicates the percentage of correct positive predictions for each gender class.
- ### Recall:

- Male: 0.97%
- Female: 0.98%
Recall shows the percentage of actual positive cases correctly identified by the model.
### F1-Score:

- Male: 0.98%
- Female: 0.98%
The F1-score is the harmonic mean of precision and recall, providing a balanced measure of the model’s accuracy.
### Confusion Matrix:
[[483  10]
 [ 14 494]]

### LIBRARY USED
- Scikit learn(sklearn)
- Pandas
- Numpy
- Matplotlib


### STEP 1
- import pandas as pd
- import numpy as np
- import matplotlib.pyplot as plt

### STEP 2
- gender=pd.read_csv("Gender_Classification Project 2.csv")
- gender

### STEP 3
#### Replacing our Gender column with 0 & 1

gender.replace("Male",1, inplace=True)
gender.replace("Female",0, inplace=True)
gender

### STEP 4
### Checking our Data type
- gender.info()
![Screenshot (244)](https://github.com/user-attachments/assets/f05dd6b1-d3a6-4b54-97e4-e00ec3805d02)

### STEP 5
# splitting our data into X and Y

- X=gender.drop(columns="gender")
- Y=gender["gender"]
- X
![Screenshot (246)](https://github.com/user-attachments/assets/30eefac0-1806-4a62-bca6-a8f414e6e3e0)


### STEP 6
### Checking our Y
- Y

![Screenshot (247)](https://github.com/user-attachments/assets/84c17f47-9961-4315-bfcb-42d39958e7a9)

### STEP 7
# SPLITTING OUR DATA INTO TEST AND TRAIN

- from sklearn.model_selection import train_test_split
- X_train, X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

### STEP 8
# testing our data with Support vector machine(SVC)

- from sklearn.svm import SVC 
- sv_model=SVC()  # you put bracket whenever you assign it to a variable

### STEP 9
### FITTING OUR DATA INTO THE MODEL
- sv_model.fit(X_train,Y_train)

### Checking the model accuracy
- sv_model.score(X_test,Y_test)

### STEP 10
# TESTING OUR DATA WITH Randomforestclassifier

- from sklearn.ensemble import RandomForestClassifier
- Rfc_model=RandomForestClassifier()

### STEP 11
- ### FITTING OUR DATA INTO THE MODEL
- Rfc_model.fit(X_train,Y_train)

### Checking our accuracy
- Rfc_model.score(X_test,Y_test)

### STEP 12
- ### PREDICTING OUR TEST DATA
- Rfc_model.predict(X_test)

### STEP 13
- ### putting our Y_test into numpy array
- np.array([Y_test])

### STEP 14
- #### PUTTING OUR PREDICTION ABOVE INTO A VARIABLE
- machine=Rfc_model.predict(X_test)

### STEP 15
- ### creating a Dataframe for the X test
- df=X_test

### STEP 16
- ### Creating a column for our result

- df['Our_result']=Y_test

### STEP 17
- ### creating a column for the machine result

- df["machine"]=machine
- df
![Screenshot (248)](https://github.com/user-attachments/assets/b73f034e-46bf-4197-abbb-2b664a4c0495)


### STEP 18
- ### Checking our Accuracy Reports 
- ### Importing Library
- from sklearn.metrics import accuracy_score

### STEP 19
- ### Getting the prediction and storing it in a variable
- y_preds=machine
- print(accuracy_score(Y_test,y_preds))

### STEP 20
- ### Classification report
- ### importing library to use
- from sklearn.metrics import classification_report


### STEP 21
- ### Getting the classfication result
- print(classification_report(Y_test,y_preds))

![Screenshot (249)](https://github.com/user-attachments/assets/458c9569-0ca6-4ec2-ac9c-739629243b7b)

### STEP 22
- ### Using Confusion Matrix
- ### importing libraries
- from sklearn.metrics import confusion_matrix
- print(confusion_matrix(Y_test,y_preds))
![Screenshot (250)](https://github.com/user-attachments/assets/0f9d8a8b-9ddc-4543-bb27-eaeb0fc91301)

### STEP 23
- ### Representing our Result Map
- ### Import library
- from sklearn.metrics import ConfusionMatrixDisplay

### STEP 24
- ### Getting results from our predictions
- plt.savefig("predictions.png")
- ConfusionMatrixDisplay.from_predictions(Y_test,y_preds)
![Screenshot (251)](https://github.com/user-attachments/assets/05a74903-a0a6-4d42-92b2-a3e07ffd2829)

### STEP 25
- ### Giving it more indepth than normal sns map, using the estimator result
- ConfusionMatrixDisplay.from_estimator(Rfc_model,X,Y)
- plt.savefig("estimator.png")
![Screenshot (252)](https://github.com/user-attachments/assets/ab804b32-0a84-4599-af93-0ea2d6c44f2e)

