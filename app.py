from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import torch

app = Flask(__name__)

# Free conversational AI model (better than GPT-2)
chatbot = pipeline(
    "text-generation",
    model="microsoft/DialoGPT-medium"
)

def generate_reply(user_message):
    prompt = f"User: {user_message}\nBot:"

    response = chatbot(
        prompt,
        max_length=100,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=50256
    )

    output = response[0]["generated_text"]

    # Extract only bot response
    reply = output.split("Bot:")[-1].strip()

    # Safety cleanup (avoid empty responses)
    if not reply:
        reply = "Sorry, I didn't understand that."

    return reply


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")

    reply = generate_reply(user_message)

    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True)