from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

data = pd.read_excel('C://Users//LENOVO//OneDrive//Desktop//Book1.xlsx')

@app.route('/')
def index():
    states = data['State'].unique().tolist()
    return render_template('index1.html', states=states)


@app.route('/predict', methods=['POST'])
def predict():
    req = request.json
    state = req['state']
    year1 = int(req['year1'])
    year2 = int(req['year2'])

    state_data = data[(data['State'] == state)]
    if state_data.empty:
        return jsonify({'error': 'State not found'}), 400

    # Predict for both years
    pred1 = model.predict([[year1]])[0]
    pred2 = model.predict([[year2]])[0]

    # Get latest known male/female ratios from data
    latest = state_data[state_data['Year'] == state_data['Year'].max()].iloc[0]
    male_ratio = latest['M_LIT'] / latest['P_LIT']
    female_ratio = latest['F_LIT'] / latest['P_LIT']

    # Male/female predictions
    male_pred1 = pred1 * male_ratio
    female_pred1 = pred1 * female_ratio
    male_pred2 = pred2 * male_ratio
    female_pred2 = pred2 * female_ratio

    # Calculate literacy change (%)
    change = (((pred2 - pred1) / pred1) * 100)

    return jsonify({
        'state': state,
        'year1': year1,
        'year2': year2,
        'predicted_lit_1': round(pred1, 2),
        'predicted_lit_2': round(pred2, 2),
        'change_percent': round(change, 2),
        'male_lit_1': round(male_pred1, 2),
        'male_lit_2': round(male_pred2, 2),
        'female_lit_1': round(female_pred1, 2),
        'female_lit_2': round(female_pred2, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)

