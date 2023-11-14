from flask import Flask, render_template, request
from emoji import emojize
from transformers import pipeline

specific_model = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")

app = Flask(__name__)

def analyze_sentiment(sentence):
    result = specific_model(sentence)
    # Extract the label with the highest confidence
    label = max(result, key=lambda x: x['score'])['label']
    return label

@app.route("/", methods=["GET", "POST"])
def opp_detector():
    response = None
    sentiment_class = None  # Add a variable to store the sentiment class

    if request.method == "POST":
        user_sentence = request.form["user_sentence"]
        sentiment_label = analyze_sentiment(user_sentence)

        if sentiment_label == "POS":
            response = "You could talk to them."
            sentiment_class = "positive"
        elif sentiment_label == "NEU":
            response = "Its a 50/50. They don't give off any warning signs but they could be untrustworthy."
            sentiment_class = "neutral"
        elif sentiment_label == "NEG":
            response = "Don't talk to them. They seem unreliable."
            sentiment_class = "negative"

    return render_template("index.html", response=response, sentiment_class=sentiment_class)

if __name__ == "__main__":
    app.run(debug=True)
