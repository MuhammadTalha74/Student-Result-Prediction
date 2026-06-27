# 🎓 Student Result Prediction System

A machine learning web application that predicts whether a student will Pass or Fail based on key academic and personal factors. Built with Python, Scikit-learn, and Flask.
---
🚀 Demo
The app takes student information as input and instantly predicts the result (Pass/Fail) through a simple web interface.
---
🧠 ML Model
Algorithm: Gaussian Naive Bayes
Task: Binary Classification (Pass / Fail)
Target: Final grade G3 — Pass if G3 ≥ 10, Fail otherwise
Training samples: 120 | Testing samples: 80
📊 Input Features
Feature	Description
`age`	Student's age
`Medu`	Mother's education level (0–4)
`Fedu`	Father's education level (0–4)
`studytime`	Weekly study time (1–4)
`failures`	Number of past class failures
---
🛠️ Tech Stack
Language: Python
ML Library: Scikit-learn
Web Framework: Flask
Data Handling: Pandas, NumPy
Visualization: Matplotlib, Seaborn
Frontend: HTML, CSS
---
📁 Project Structure
```
Student-Result-Prediction/
│
├── project/
│   ├── app.py                  # Flask web application
│   ├── model.py                # ML model training script
│   ├── student_data.csv        # Dataset
│   ├── model.pkl               # Saved trained model
│   ├── static/
│   │   └── confusion_matrix.png
│   └── Templates/
│       ├── Index.html          # Input form
│       └── result.html         # Prediction result page
│
└── Screenshots/                # App screenshots
```
---
⚙️ How to Run
1. Clone the repository
```bash
git clone https://github.com/MuhammadTalha74/Student-Result-Prediction.git
cd Student-Result-Prediction/project
```
2. Install dependencies
```bash
pip install flask scikit-learn pandas numpy matplotlib seaborn
```
3. Train the model (generates model.pkl)
```bash
python model.py
```
4. Run the Flask app
```bash
python app.py
```
5. Open in browser
```
http://127.0.0.1:5000
```
---
📸 Screenshots
> Add your screenshots from the `Screenshots/` folder here.
---
👨‍💻 Author
Muhammad Talha  
Final-year Software Engineering Student  
GitHub • LinkedIn
