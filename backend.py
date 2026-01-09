import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import time
from IPython import display
from sklearn.preprocessing import StandardScaler
from flask import Flask, jsonify, request
from sklearn.datasets import make_classification

app = Flask(__name__)

X, y = make_classification(
    n_samples=100,
    n_features=2,           # 2D data (two columns in X)
    n_informative=2,        # Both features contribute to the target
    n_redundant=0,          # No linear combinations of other features
    n_clusters_per_class=1, # Simple single-blob distribution for each class
    random_state=42
)
df = pd.DataFrame(X, columns=['coordinate X', 'coordinate Y'])

@app.route('/display', methods = ['GET'])
def display_data():
    return jsonify({
        "data": df.to_dict('records')
    })

#standardizing the data
x_train,x_test,y_train,y_test = train_test_split(X,y)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
x_train = np.insert(x_train,0,1,axis = 1)
x_test = np.insert(x_test,0,1,axis = 1)

#logistic reg using gradient descent
class gradient_descent:
    def __init__(self,x_train,y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.B = np.zeros((x_train.shape[1],1))
    def fit(self,epochs,lr,no_batches):
        print(no_batches)
        num = self.x_train.shape[0] // no_batches
        slopes_array = []
        for j in range(epochs):
            val1 = 0
            for i in range(no_batches):   # it was rigth
                val2 = val1 + num 
                batch_x = self.x_train[val1:val2]   #here i only did for x , i have to do also for y
                batch_y = self.y_train[val1:val2]
                batch_y = batch_y.reshape(len(batch_y),1)
                val1 = val2 
                    #here batch becomes std_xtrain
                z = np.dot(batch_x,self.B)
                    #print(np.shape(z))
                y_pred = 1/ (1 + np.exp(-z))
                    #print(np.shape(y_pred))
                    #print(np.shape(batch_y))
                    #print(np.shape(batch_x))
                slope = np.dot(batch_x.T,y_pred - batch_y )/no_batches
                self.B = self.B - lr*slope
                slopes_array.append(self.B.reshape(x_train.shape[1],))
        return self.B,slopes_array
    def predict(self,x_data):
        z = np.dot(x_data,self.B)
        y_pred = 1/ (1 + np.exp(-z))    #just interchanging position of two matrices
        return (y_pred)

gd = gradient_descent(x_train,y_train)
'''if GD == 'batch GD':

    B,slopes_array = gd.fit(ep,lr,no_batches)
    y_pred = gd.predict(x_test)
    print(y_pred)
        #error = gd.vis(slopes_array)
elif GD == 'mini-batch GD':
    val = int(input('Enter the number of batches : '))
    B,slopes_array = gd.fit(ep,lr,no_batches)
    y_pred = gd.predict(x_test)
        #error = gd.vis(slopes_array)
elif GD == 'stochastic GD':
    no_batches = x_train.shape[0]
    B,slopes_array = gd.fit(ep,lr,no_batches )
    y_pred = gd.predict(x_test)
        #error = gd.vis(slopes_array)
    #error = (y_pred - y_test)**2
print(B)'''

# Generate synthetic data for classification

@app.route('/train', methods = ['POST'])
def train_model():
    data = request.json   # ✅ ONLY hyperparameters
    mode = data.get("mode", "batch")
    gd = gradient_descent(x_train,y_train)
    lr = float(data.get("lr", 0.01))
    epochs = int(data.get("epochs", 10))

    if mode == 'batch':
        batch_size = 1
        B,slopes_array = gd.fit(epochs,lr,batch_size)
        y_pred = gd.predict(x_test)
    elif mode == 'mini-batch':
        batch_size = int(data.get("batch_size", 5))
        B,slopes_array = gd.fit(epochs,lr,batch_size)
        y_pred = gd.predict(x_test)
    elif mode == 'stochastic':
        batch_size = x_train.shape[0]
        B,slopes_array = gd.fit(epochs,lr,batch_size)
        y_pred = gd.predict(x_test)
    return jsonify({
        "message": "Model trained ho gaya! 👍",
        "coefficients": B.flatten().tolist(),
        "slopes_array": [s.flatten().tolist() for s in slopes_array],
        "y_pred": y_pred.flatten().tolist(),
        "batch_size": batch_size
    }), 200

if __name__ == '__main__':
    app.run(debug=True,port=5001)
