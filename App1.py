from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

#load the data set
iris = load_iris()
X = iris.data
y = iris.target

# 20% for testing and 80%for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
# random forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

import pickle
pickle.dump(model, open('model.pickle','wb'))

from flask import Flask,jsonify,request
app=Flask(__name__)
model=pickle.load(open('model.pickle','rb'))

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict/',methods=['POST'])
def class_predict():
    int_features=[int(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
    return render_template('index.html',prediction_text='The species of iris is ${}'.format(prediction))

if __name__=='__main__':
    app.run(debug=True)

