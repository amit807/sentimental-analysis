import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from sklearn import svm
from sklearn.metrics import classification_report
import pickle
import joblib
from textblob import TextBlob


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
vectorizer = joblib.load('model_vectorizer.pkl')


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=["GET","POST"])
def predict():

        text = request.form.get("inp")
        #text = text.lower()
        analysisPol = TextBlob(text).polarity
        feature = vectorizer.transform([text])
        score = model.predict(feature)

        #output= print("The polarity value is : ",analysisPol\n)
        #return render_template("home.html", message="The polarity value is :{}".format(analysisPol))
        #return render_template("home.html", message1="Therefore, the review is {}".format(score))
        #print("The polarity value for the review is :", score)
        if score == "Negative":
            return render_template("home.html", message="Therefore the review is NEGATIVE with a polarity value of {}".format(analysisPol))
        elif score == "Positive":
            return render_template("home.html", message="Therefore the review is POSITIVE with a polarity value of {}".format(analysisPol))
        else:
            return render_template("home.html")

    
if __name__ == "__main__":
    app.run(debug=True)