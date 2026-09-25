from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

import webbrowser
from threading import Timer


application = Flask(__name__)

# Keep "app" as an alias
app = application


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():

    if request.method == "GET":
        return render_template("home.html")

    try:
        # Get data from HTML form
        data = CustomData(
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("ethnicity"),
            parental_level_of_education=request.form.get(
                "parental_level_of_education"
            ),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get(
                "test_preparation_course"
            ),
            reading_score=float(
                request.form.get("reading_score")
            ),
            writing_score=float(
                request.form.get("writing_score")
            ),
        )

        # Convert data into DataFrame
        pred_df = data.get_data_as_data_frame()

        # Make prediction
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # Ensure a clean plain number reaches the template
        # (results[0] can come back as a numpy type)
        predicted_score = round(float(results[0]), 2)

        return render_template(
            "home.html",
            results=predicted_score,
            form_data=request.form
        )

    except Exception as e:

        print(f"Prediction error: {e}")

        return render_template(
            "home.html",
            results="Please review the information and try again.",
            form_data=request.form
        )


# Run application
if __name__ == "__main__":

    # Browser URL
    url = "http://127.0.0.1:5000"

    # Open browser automatically after 1 second
    Timer(
        1,
        lambda: webbrowser.open(url)
    ).start()

    # Start Flask server
    app.run(
        host="0.0.0.0",
        port=5000
    )
