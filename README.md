# Happiness Score Predictor (ML + Flask)

A Machine Learning web application that predicts the **Happiness Score of a country** based on socio-economic factors.

---

##  Live Demo
https://happiness-score-predictor-zo2p.onrender.com/
---

##  Features

- Predicts happiness score using ML model
- User-friendly web interface
- Real-time prediction
- Based on multiple socio-economic factors:
  - Economy (GDP per capita)
  - Family support
  - Health (Life expectancy)
  - Freedom
  - Trust (Government corruption)
  - Generosity

---

## How It Works

1. User inputs feature values in the web form  
2. Data is converted into numerical format  
3. Model processes input features  
4. Prediction is generated and displayed  

---

##  Tech Stack

- Python
- Flask
- Scikit-learn
- Pandas, NumPy
- HTML, CSS

---

##  Machine Learning Details

- Dataset: World Happiness Report (2015)
- Models Tried:
  - Decision Tree
  - Random Forest
  - XGBoost
  - Linear Regression  (Selected)

- Evaluation Metrics:
  - R² Score
  - Mean Squared Error

---


---

## Run Locally

pip install -r requirements.txt

python app.py
