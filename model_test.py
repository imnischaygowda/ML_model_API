
import numpy as np
import requests
import json 

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# load data
iris = load_iris()

# split into train and test sets using same random state
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=42)

# Serialize the data into json and send th request to model.
payload = {'data': json.dumps(X_test.tolist())}
y_predict = requests.post('http://127.0.0.1:5000/iris', data=payload).json()

# Make array from list
y_predict = np.array(y_predict)
print(y_predict)
