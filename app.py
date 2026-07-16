from flask import Flask, render_template, request
import pandas as pd
import os

from sentiment import predict_sentiment
from visualize import create_chart

app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def home():

    reviews = []
    sentiments = []

    if request.method == "POST":

        file = request.files["file"]

        df = pd.read_csv(file)

        for review in df["Review"]:

            label, score = predict_sentiment(review)

            label = label.lower()

            reviews.append({
                "review": review,
                "sentiment": label,
                "score": score
            })

            sentiments.append(label)

        if not os.path.exists("static"):
            os.mkdir("static")

        create_chart(sentiments)

    return render_template("index.html",
                           reviews=reviews)

if __name__ == "__main__":
    app.run(debug=True)