import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pickle
import utiles

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model_1.pkl", "rb"))
@flask_app.route("/")
def Home():
    return render_template("index.html")
@flask_app.route("/service")
def service():
    return render_template("matrix_services.html")

@flask_app.route("/about")
def about():
    return render_template("matrix_about_us.html")

@flask_app.route("/contact")
def contact():
    return render_template("contact.html")


@flask_app.route("/predict", methods = ["GET","POST"])
def predict():
    
    if request.method=='GET':
        return render_template('matrix_services.html')
    else:
        Age = request.form.get('Age')
        Gender = request.form.get('Gender')
        Stream = request.form.get('Stream')
        Intership = request.form.get('Intership')
        CGPA = request.form.get('CGPA')
        Backlog = request.form.get('Backlog')
        Hostel = request.form.get('Hostel')

        result = utiles.preprocess(Age,Gender,Stream,Intership,CGPA,Backlog,Hostel)
        print(result)
        # print(data)

        if(result == 1):
            return render_template('matrix_output01.html',result=result)
        else: 
            return render_template('matrix_output02.html',result=result)    
        # return render_template('matrix_services.html',result=result)


if __name__ == "__main__":
    flask_app.run(debug=True)