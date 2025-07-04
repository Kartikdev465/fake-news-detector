import os
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Safe file loading using absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, 'fake_news_model.pkl')
vectorizer_path = os.path.join(BASE_DIR, 'vectorizer.pkl')

# Load model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Load vectorizer
with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['news']
    transformed = vectorizer.transform([input_text]).toarray()
    prediction = model.predict(transformed)
    result = 'Real News ✅' if prediction[0] == 1 else 'Fake News ❌'
    return render_template('index.html', prediction=result)

# FINAL production-ready server block
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
