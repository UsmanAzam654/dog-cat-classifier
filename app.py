from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model("model.h5")

IMG_SIZE = (150, 150)

@app.route('/')
def home():
    return "API is running"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"})

    file = request.files["file"]

    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    img = img.resize(IMG_SIZE)

    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]

    result = "Dog 🐶" if prediction > 0.5 else "Cat 🐱"

    return jsonify({"prediction": result})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)