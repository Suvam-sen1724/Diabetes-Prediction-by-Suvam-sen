import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

def predict_diabetes():
    input_data = np.array([[
        float(entry1.get()),
        float(entry2.get()),
        float(entry3.get()),
        float(entry4.get()),
        float(entry5.get()),
        float(entry6.get()),
        float(entry7.get()),
        float(entry8.get())
    ]])
    
    # Standardize the input data
    std_data = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(std_data)
    
    if prediction[0] == 0:
        messagebox.showinfo('Prediction', 'The person is not diabetic')
    else:
        messagebox.showinfo('Prediction', 'The person is diabetic')

# Create a tkinter window
root = tk.Tk()
root.geometry("600x600")
root.configure(bg='grey')

# Create text boxes for user input with labels and spacing
labels = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
entries = []

for i, label in enumerate(labels):
    tk.Label(root, text=label, bg='black', fg='white').pack(pady=5)
    entry = tk.Entry(root)
    entry.pack(pady=5)
    entries.append(entry)

entry1, entry2, entry3, entry4, entry5, entry6, entry7, entry8 = entries

# Create a 'Predict' button
button = tk.Button(root, text='Predict', command=predict_diabetes)
button.pack(pady=10)

root.mainloop()
