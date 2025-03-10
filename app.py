from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
model_pipeline = pickle.load(open("E:\Data Science\Data Science Mini Projects\Cricket-Score-Prediction\model\cricket_score_prediction.pkl", "rb"))

# Dropdown options
teams = [
    'Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa',
    'England', 'West Indies', 'Afghanistan', 'Pakistan', 'Sri Lanka'
]
cities = [
    'Colombo', 'Mirpur', 'Johannesburg', 'Dubai', 'Auckland', 'Cape Town',
    'London', 'Pallekele', 'Barbados', 'Sydney', 'Melbourne', 'Durban',
    'St Lucia', 'Wellington', 'Lauderhill', 'Hamilton', 'Centurion',
    'Manchester', 'Abu Dhabi', 'Mumbai', 'Nottingham', 'Southampton',
    'Mount Maunganui', 'Chittagong', 'Kolkata', 'Lahore', 'Delhi',
    'Nagpur', 'Chandigarh', 'Adelaide', 'Bangalore', 'St Kitts', 'Cardiff',
    'Christchurch', 'Trinidad'
]

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        data = request.form
        form_data = extract_form_data(data)
        input_data = prepare_input_data(form_data)
        prediction = model_pipeline.predict(input_data)[0]
        prediction = round(prediction)
    
    return render_template("index.html", teams=teams, cities=cities, prediction=prediction)

def extract_form_data(data):
    return {
        'batting_team': data['batting_team'],
        'bowling_team': data['bowling_team'],
        'city': data['city'],
        'current_score': int(data['current_score']),
        'balls_left': int(data['balls_left']),
        'wicket_left': int(data['wicket_left']),
        'last_five': int(data['last_five'])
    }

def prepare_input_data(form_data):
    balls_bowled = 120 - form_data['balls_left']
    current_run_rate = (form_data['current_score'] * 6) / balls_bowled if balls_bowled > 0 else 0
    form_data['current_run_rate'] = current_run_rate
    return pd.DataFrame([form_data])

if __name__ == "__main__":
    app.run(debug=True)
