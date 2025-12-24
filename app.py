from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])

        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        result = model.predict(features)[0]

        species = ["Setosa", "Versicolor", "Virginica"]
        prediction = species[result]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
