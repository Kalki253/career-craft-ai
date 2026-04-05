import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)
# Enable CORS for all routes (to simplify frontend integrations)
CORS(app, resources={r"/*": {"origins": "*"}})

# ---- Mock Machine Learning Model Setup ----
# Since we need to predict careers based on Interest + Skill + Stream,
# we will create a basic dataset and train a simple DecisionTreeClassifier in memory.

TRAINING_DATA = [
    # [interest_code, skill_code, stream_code]
    # Streams: 1: Science, 2: Commerce, 3: Arts
    # Skills: 1: Math, 2: Analysis, 3: Communication, 4: Creativity
    # Interests: 1: Coding, 2: Biology, 3: Business, 4: Design
    
    # coding + math + science -> Software Engineer
    {"interest": 1, "skill": 1, "stream": 1, "career": "Software Engineer"},
    
    # biology + analysis + science -> Doctor
    {"interest": 2, "skill": 2, "stream": 1, "career": "Doctor"},
    
    # business + communication + commerce -> Entrepreneur
    {"interest": 3, "skill": 3, "stream": 2, "career": "Entrepreneur"},
    
    # design + creativity + arts -> UI/UX Designer
    {"interest": 4, "skill": 4, "stream": 3, "career": "UI/UX Designer"}
]

df = pd.DataFrame(TRAINING_DATA)
X = df[["interest", "skill", "stream"]]
y = df["career"]

# Train the model
model = DecisionTreeClassifier()
model.fit(X, y)

# Helper dictionaries to encode incoming JSON text inputs to numeric codes used by the model
INTERESTS_MAP = {"coding": 1, "biology": 2, "business": 3, "design": 4}
SKILLS_MAP = {"math": 1, "analysis": 2, "communication": 3, "creativity": 4}
STREAMS_MAP = {"science": 1, "commerce": 2, "arts": 3}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "No input provided"}), 400

        interest_raw = str(data.get("interest", "")).lower().strip()
        skill_raw = str(data.get("skill", "")).lower().strip()
        stream_raw = str(data.get("stream", "")).lower().strip()

        if not interest_raw or not skill_raw or not stream_raw:
            return jsonify({"success": False, "message": "Missing interest, skill, or stream parameters"}), 400

        # Encode (fallback to some default like science/coding if unknown just for the project functionality)
        i_code = INTERESTS_MAP.get(interest_raw, 1)
        k_code = SKILLS_MAP.get(skill_raw, 1)
        s_code = STREAMS_MAP.get(stream_raw, 1)

        # Predict
        pred = model.predict([[i_code, k_code, s_code]])
        career = pred[0]

        return jsonify({"success": True, "career": career})

    except Exception as e:
        print("ML Prediction Error:", str(e))
        return jsonify({"success": False, "message": "Internal AI server error."}), 500

if __name__ == "__main__":
    # For deployment ensure this binds to 0.0.0.0
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
