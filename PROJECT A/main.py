from flask import Flask, render_template, request, redirect, url_for 
import joblib
import numpy as np

app = Flask(__name__)
loaded_model= joblib.load(r'C:\Users\hanin\OneDrive\Desktop\DS\PROJECT_Hanin Ismail\PROJECT A\model.pkl')

@app.route("/")
def root():
    return render_template("index.html")

@app.route("/PROJECT A", methods= ["POST"])
def make_prediction():
    if request.method == "POST":
        ID= request.form['ID']
        Age = request.form['Age']
        freq = request.form['freq']
        female = request.form['female']
        male = request.form['male']
        
        x = np.array([int(ID), int(Age), int(freq),int(female), int(male)]).reshape(1,-1)
        #x = [[int(ID), int(Age), int(freq),int(female), int(male)]]
        print(f"x type: {type(x)}")
        
        prediction = loaded_model.predict(x)
        print(f"prediction type: {type(prediction)}")
        
        predictedClass = prediction
        
        msg = f"The predicted class is {predictedClass}"
        return render_template("index.html", prediction_text = msg )
    
if __name__ == "__main__":
    app.run(debug =True)
        
       # mkvirtualenv --python=/usr/bin/python3.6 my-virtualenv 