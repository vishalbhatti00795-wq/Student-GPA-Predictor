# 🎓 Student GPA Predictor
[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-View%20App-success?style=for-the-badge)](https://student-gpa-predictor.onrender.com)

A simple Machine Learning project that predicts a student's **GPA out of 4.0** based on their study habits, attendance, and other academic activities.

The project uses **KNN Regression** and provides an interactive web interface using **Streamlit**.

## 🚀 Features

- Predicts student GPA
- Uses KNN Regression
- Uses StandardScaler for feature scaling
- Simple and interactive Streamlit UI
- Shows GPA and performance level

## 🧠 Machine Learning

**Algorithm:** K-Nearest Neighbors (KNN) Regression  
**K:** 4  
**Features:** 8  
**Scaler:** StandardScaler  

The model predicts GPA by looking at students with similar characteristics.

## 📊 Input Features

The prediction is based on:

- Weekly Study Time
- Number of Absences
- Grade Class
- Tutoring
- Parental Support
- Extracurricular Activities
- Sports
- Music

## 🛠️ Technologies Used

- Python
- Scikit-learn
- NumPy
- Streamlit

## 📂 Project Structure

```text
Student-GPA-Predictor/
│
├── app.py
├── model.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```
