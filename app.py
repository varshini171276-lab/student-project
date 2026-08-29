from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = "model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        study_hours = float(request.form["study_hours"])
        attendance = float(request.form["attendance"])
        sleep_hours = float(request.form["sleep_hours"])
        previous_marks = float(request.form["previous_marks"])

        data = np.array([[study_hours, attendance, sleep_hours, previous_marks]])

        if model is not None:
            score = model.predict(data)[0]
        else:
            score = (study_hours * 5) + (attendance * 0.2) + (sleep_hours * 2) + (previous_marks * 0.5)

        result = "Pass" if score >= 60 else "Fail"
        prediction = f"Predicted Final Score: {score:.1f} | Result: {result}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
