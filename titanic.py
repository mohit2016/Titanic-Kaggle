# -*- coding: utf-8 -*-


#Importing the libraries
import numpy as np
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, [2,4,5,6,7,9]].values
y = dataset.iloc[: , 1].values

#Taking care of missing dataset
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values ="NaN", strategy ="mean", axis =0)
imputer = imputer.fit(X[:,0:6])
X[:,0:6] = imputer.transform(X[:,0:6])

#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,1] = labelencoder_X.fit_transform(X[:,1])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)




import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#define nerual network
classifier = Sequential()

#add first layer and input layer
classifier.add(Dense(units=4, kernel_initializer="uniform",activation="relu", input_dim=6))

#add second hidden layer
classifier.add(Dense(units=4, kernel_initializer="uniform",activation="relu"))

classifier.add(Dense(units=4, kernel_initializer="uniform",activation="relu"))

classifier.add(Dense(units=4, kernel_initializer="uniform",activation="relu"))

#add ouput layer
classifier.add(Dense(units=1, kernel_initializer="uniform",activation="sigmoid"))
#compile ANN
classifier.compile(optimizer="rmsprop", loss="binary_crossentropy",metrics=["accuracy"])
#fitting ANN
classifier.fit(X,y,batch_size=25,epochs=100)






#PART- TUNING Parameters
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
def build_classifier(optimizer):
    cl = Sequential()
    cl.add(Dense(units=4,kernel_initializer="uniform",activation="relu",input_dim=6))
    cl.add(Dropout(rate=0.2))
    cl.add(Dense(units=4,kernel_initializer="uniform",activation="relu"))
    cl.add(Dropout(rate=0.2))
    cl.add(Dense(units=4,kernel_initializer="uniform",activation="relu"))
    cl.add(Dropout(rate=0.2))
    cl.add(Dense(units=1,kernel_initializer="uniform",activation="relu"))
    cl.compile(optimizer=optimizer,loss="binary_crossentropy", metrics=["accuracy"])
    return cl

classifier = KerasClassifier(build_fn=build_classifier)

parameters= {
        'batch_size': [25,32],
        'epochs':[100,200],
        'optimizer':["adam","rmsprop"]
        }
grid_search = GridSearchCV(estimator=classifier,param_grid=parameters,scoring="accuracy",cv=10)
grid_search =grid_search.fit(X,y)
print(grid_search.best_score_)
best_params = grid_search.best_params_
#best params:-
# batch_size=25
# epochs=100
#optimizer=rmsprop





#Importing real testset
test_data = pd.read_csv("test.csv")
X_test = test_data.iloc[:,[1,3,4,5,6,8]].values

#Taking care of missing dataset
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values ="NaN", strategy ="mean", axis =0)
imputer = imputer.fit(X_test[:,0:6])
X_test[:,0:6] = imputer.transform(X_test[:,0:6])

#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
X_test[:,1] = labelencoder_X1.fit_transform(X_test[:,1])
onehotencoder = OneHotEncoder(categorical_features = [1])
X_test = onehotencoder.fit_transform(X_test).toarray()
X_test = X_test[:,1:]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X1 = StandardScaler()
X_test = sc_X.fit_transform(X_test)





y_pred = classifier.predict(X_test)


#Boolean format for y_pred
y_pred = y_pred>0.5
a=[]
for i in range(0,418):
    if y_pred[i]:
        a.append(1)
    else:
        a.append(0)