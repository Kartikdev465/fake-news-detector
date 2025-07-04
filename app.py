from flask import Flask, render_template, request
import pickle

# Initialize app
app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open('fake_news_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_text = request.form['news']
        transformed_input = vectorizer.transform([input_text]).toarray()
        prediction = model.predict(transformed_input)

        result = 'Real News ✅' if prediction[0] == 1 else 'Fake News ❌'
        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
