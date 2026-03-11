from flask import Flask, render_template, request
import numpy as np
import pickle
import os

# Load model
model = pickle.load(open("bestmodel.pkl","rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/prediction")
def prediction():
    return render_template("prediction.html")

@app.route("/predict", methods=["POST"])
def predict():

    economy = float(request.form["economy"])
    family = float(request.form["family"])
    health = float(request.form["health"])
    freedom = float(request.form["freedom"])
    trust = float(request.form["trust"])
    generosity = float(request.form["generosity"])

    features = np.array([[economy, family, health, freedom, trust, generosity]])

    result = model.predict(features)

    return render_template("result.html", prediction=round(result[0],3))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
