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
worksheet = gc.open('ex1').sheet1
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



<img width="149" alt="dl-1" src="https://github.com/charansai0/basic-nn-model/assets/94296221/b0c39743-9a95-4536-86c2-e83b83e1858a">


## OUTPUT

### Training Loss Vs Iteration Plot



<img width="359" alt="dl-g" src="https://github.com/charansai0/basic-nn-model/assets/94296221/aca785b4-8dbe-4490-adfb-7c1274fec888">


### Test Data Root Mean Squared Error
<img width="323" alt="dl-2" src="https://github.com/charansai0/basic-nn-model/assets/94296221/9c299477-b826-4a3f-8278-a9026bd525b4">



### New Sample Data Prediction


<img width="239" alt="dl-3" src="https://github.com/charansai0/basic-nn-model/assets/94296221/c4c98b74-0da6-4248-b124-741911647863">


## RESULT
Thus the neural network regression model for the given dataset is executed successfully.
