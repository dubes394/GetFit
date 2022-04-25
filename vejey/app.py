import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

import joblib 

from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

global loaded_model
df = pd.read_csv("Iris.csv")
X=df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y=df[['Species']]

X_train,X_test,y_train,y_test = train_test_split(X,
                                                y,
                                                test_size=0.2,
                                                random_state=0
)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,y_train)

model_filename = "My_KNN_Model.model"

joblib.dump(model, model_filename)

loaded_model = joblib.load(model_filename)
predict1 = loaded_model.predict(X_test)
print("The accuracy of the KNN Classifier is: ",metrics.accuracy_score(predict1, y_test))

loaded_model = joblib.load("My_KNN_Model.model")
# GCS bucket
# AWS S3 bucket
# Azure Blob Storage

@app.route('/')
def home():
    # return "<h1>Welcome to Iris Classifier. Use '/classify' route to classify an iris sample. </h1>"
    return render_template('index.html')

@app.route("/train")
def train():
    global loaded_model
    df = pd.read_csv("Iris.csv")
    X=df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    y=df[['Species']]

    X_train,X_test,y_train,y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=0
    )

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train,y_train)

    model_filename = "My_KNN_Model.model"

    joblib.dump(model, model_filename)

    loaded_model = joblib.load(model_filename)
    predict1 = loaded_model.predict(X_test)
    print("The accuracy of the KNN Classifier is: ",metrics.accuracy_score(predict1, y_test))
    return {
        'code': 200,
        'message': 'Model is created.'
    }


# 1. Train a global model outside/inside the flask application and save the trained model
# At inference time, you need to load the model and predict on the given sample

# Train the model with /train endpoint (save the trained model)
# At inference time, you need to load the model and predict on the given sample

@app.route("/classify")
def classify():
    loaded_model = joblib.load("My_KNN_Model.model")
    # 2. Classify the sample with the trained model
    data = { 'SepalLengthCm' : [5.8], 
            'SepalWidthCm': [2.8], 
            'PetalLengthCm': [5.1], 
            'PetalWidthCm': [2.4]
    }
    sample = pd.DataFrame(data, columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])

    class_label = loaded_model.predict(sample)
    return {
        'code': 200,
        'message': f'Sample is classified as {class_label[0]}',
        'class': class_label[0]
    }

# 1. Get rid of Harcoded parameters
# 2. 
@app.route('/classifier')
def classifier():
    sepalLength = request.args['sepalLength']
    sepalWidth = request.args['sepalWidth']
    petalLength = request.args['petalLength']
    petalWidth = request.args['petalWidth']

    data = { 'SepalLengthCm' : [sepalLength], 
            'SepalWidthCm': [sepalWidth], 
            'PetalLengthCm': [petalLength], 
            'PetalWidthCm': [petalWidth]
    }
    sample = pd.DataFrame(data, columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])

    class_label = loaded_model.predict(sample)
    return {
        'code': 200,
        'message': f'Sample is classified as {class_label[0]}',
        'class': class_label[0]
    }
    

@app.route('/final_classifier', methods=['GET', 'POST'])
def final_classifier():
    sepalLength = float(request.form.get('sepalLength'))
    sepalWidth = float(request.form.get('sepalWidth'))
    petalLength = float(request.form.get('petalLength'))
    petalWidth = float(request.form.get('petalWidth'))

    data = { 'SepalLengthCm' : [sepalLength], 
            'SepalWidthCm': [sepalWidth], 
            'PetalLengthCm': [petalLength], 
            'PetalWidthCm': [petalWidth]
    }
    sample = pd.DataFrame(data, columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])

    class_label = loaded_model.predict(sample)
    
    return render_template('result.html', prediction=class_label[0])
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', port='5000', debug=True)