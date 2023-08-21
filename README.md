# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
~~~
Developed By: Charan sai.V
Reference Number : 212221240061
~~~
### Importing Required Packages :
~~~
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
~~~
### Authentication and Creating DataFrame From DataSheet :
~~~
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('StudentsData').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'INPUT':'float'})
df = df.astype({'OUTPUT':'float'})
df
~~~
### Assigning X and Y values :
~~~
X = df[['INPUT']].values
Y = df[['OUTPUT']].values
~~~
### Normalizing the data :
~~~
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.33,random_state=33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)
~~~
### Creating and Training the model :
~~~
model = Sequential([
    Dense(5,activation = 'relu'),
    Dense(10,activation = 'relu'),
    Dense(1)
])
model.compile(optimizer='rmsprop',loss = 'mse')
model.fit(X_train1,y_train,epochs=2200)
~~~
### Plot the loss :
~~~
loss_df = pd.DataFrame(model.history.history)
loss_df.plot()
~~~
### Evaluate the Model :
~~~
X_test1 = Scaler.transform(X_test)
model.evaluate(X_test1,y_test)
~~~
### Prediction for a value :
~~~
X_n1 = [[20]]
X_n1_1 value = Scaler.transform(X_n1)
model.predict(X_n1_1 value)
~~~

## Dataset Information
<img width="206" alt="261372674-7d583125-408c-4747-afcb-b2ed31ac29d0" src="https://github.com/charansai0/basic-nn-model/assets/94296221/c13909f2-d804-4de0-9146-52fd70156b79">



## OUTPUT

### Training Loss Vs Iteration Plot


<img width="416" alt="261373196-d6bf7fe9-a9d6-4dd5-a90f-da1b560d9cc1" src="https://github.com/charansai0/basic-nn-model/assets/94296221/6d742fbd-7ba2-48f9-8e1c-8a86ebfd6c99">


### Test Data Root Mean Squared Error
<img width="410" alt="261373610-fdff8d7f-65d5-4648-963f-ab3a9e9b80e2" src="https://github.com/charansai0/basic-nn-model/assets/94296221/e48191a1-632f-40b5-a59c-1657e56a6fd1">


### New Sample Data Prediction
<img width="344" alt="261373908-38cbb6a6-7224-41bd-8863-799479da9e35" src="https://github.com/charansai0/basic-nn-model/assets/94296221/876dda7c-0718-4c9b-8011-d260f5bd9f2d">



## RESULT
Thus the neural network regression model for the given dataset is executed successfully.
