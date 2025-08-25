# ================== Flask App for Crime Prediction ==================
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# ---------------- STEP 1: Load models and features ----------------
with open("closure_model.pkl", "rb") as f:
    closure_model = pickle.load(f)

with open("police_model.pkl", "rb") as f:
    police_model = pickle.load(f)

with open("features_closure.pkl", "rb") as f:
    closure_features = pickle.load(f)

with open("features_police.pkl", "rb") as f:
    police_features = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# ---------------- STEP 2: Preprocess Input ----------------
def preprocess_input(data, feature_list):
    """
    Convert input data into the same format as training features
    """
    df = pd.DataFrame([data])

    # Apply label encoders
    for col, le in label_encoders.items():
        if col in df.columns:
            try:
                df[col] = le.transform(df[col].astype(str))
            except:
                # If unseen category → assign 0
                df[col] = 0

    # Reindex to match training features
    df = df.reindex(columns=feature_list, fill_value=0)
    return df

# ---------------- STEP 3: Routes ----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        year = int(request.form.get("year"))
        city = request.form.get("city")
        crime = request.form.get("crime")
        victim_gender = request.form.get("victim_gender", "Unknown")
        age_bracket = request.form.get("age_bracket", "20-30")

        # Build input dict
        input_data = {
            "year": year,
            "city": city,
            "crime_description": crime,
            "Victim Gender": victim_gender,
            "age_bracket": age_bracket
        }

        # Preprocess
        df_closure = preprocess_input(input_data, closure_features)
        df_police = preprocess_input(input_data, police_features)

        # Predictions
        closure_pred = closure_model.predict(df_closure)[0]
        police_pred = police_model.predict(df_police)[0]

        return jsonify({
            "case_closure_prediction": int(closure_pred),
            "police_deployment_prediction": int(police_pred)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ---------------- Run App ----------------
if __name__ == "__main__":
    app.run(debug=True)
