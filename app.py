from flask import Flask, render_template, request
import numpy as np
import joblib
from sklearn.metrics import accuracy_score
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load('model.pkl')

# Load sample test data to calculate accuracy
df = pd.read_csv('creditcard.csv')
df = df.drop(['Time'], axis=1)
X = df.drop('Class', axis=1)
y = df['Class']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X[['Amount']])

# Calculate model accuracy (note: this is approximate)
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
accuracy_percent = round(accuracy * 100, 2)

@app.route('/')
def index():
    return render_template('index.html', accuracy=accuracy_percent)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [float(request.form[f'V{i}']) for i in range(1, 29)]
        amount = float(request.form['Amount'])
        input_data.append(amount)
        final_data = np.array([input_data])
        prediction = model.predict(final_data)[0]
        result = "🚨 Fraudulent" if prediction == 1 else "✅ Legitimate"
        return render_template('index.html', prediction_text=f"Transaction is: {result}", accuracy=accuracy_percent)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
