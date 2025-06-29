from flask import Flask, render_template, request, jsonify, redirect, url_for
import joblib
import os
import pandas as pd
import requests
from urllib.parse import quote  

TOGETHER_API_KEY = "ca833d6e3454b4cd2cc3415bda487929b8ae5e20420bfda9937a19d4f07244cc"

app = Flask(__name__)

# Load models
MODEL_DIR = "models"
rf_dep = joblib.load(os.path.join(MODEL_DIR, "rf_depression_model.pkl"))
rf_anx = joblib.load(os.path.join(MODEL_DIR, "rf_anxiety_model.pkl"))
le_dep = joblib.load(os.path.join(MODEL_DIR, "le_depression.pkl"))
le_anx = joblib.load(os.path.join(MODEL_DIR, "le_anxiety.pkl"))

# Columns
phq_cols = [f'PHQ9_Item_{i}' for i in range(1, 10)]
gad_cols = [f'GAD7_Item_{i}' for i in range(1, 8)]
demo_cols = [
    'Age_Group', 'Gender', 'Employment_Status', 'Sleep_Hours',
    'Sleep_Quality', 'Physical_Activity', 'Social_Interaction',
    'Substance_Use', 'Mood_Rating', 'Stress_Level',
    'Chronic_Health_Conditions', 'Medication_Usage'
]

def generate_llama_conversation(data, pred_dep, pred_anx, tone):
    try:
        phq_score = sum(data[f] for f in phq_cols)
        gad_score = sum(data[f] for f in gad_cols)

        summary = f"""
        Age Group: {data.get('Age_Group')}
        Gender: {data.get('Gender')}
        Employment: {data.get('Employment_Status')}
        PHQ-9 score: {phq_score}
        GAD-7 score: {gad_score}
        Depression Prediction: {pred_dep}
        Anxiety Prediction: {pred_anx}
        Sleep Hours: {data.get('Sleep_Hours')}
        Sleep Quality: {data.get('Sleep_Quality')}
        Stress Level: {data.get('Stress_Level')}
        Mood Rating: {data.get('Mood_Rating')}
        Social Interaction: {data.get('Social_Interaction')}
        Physical Activity: {data.get('Physical_Activity')}
        Chronic Conditions: {data.get('Chronic_Health_Conditions')}
        Medication: {data.get('Medication_Usage')}
        """
       
        tone_instructions = {
            "Supportive": "Speak warmly, gently, and compassionately like a licensed therapist..",
            "Motivational": "Use an uplifting and encouragement with emotional intelligence like a licensed therapist.",
            "Direct": "Speak as a clinical psychiatrist: focused, respectful, and brief."
        }


        prompt = (
            f"{tone_instructions.get(tone, '')}\n"
            f"You are a compassionate mental health assistant.\n"
            f"The user's intake form data is:\n{summary.strip()}\n"
            f"Please start a therapeutic conversation to:\n"
            f"1. Gently validate the predicted severity levels.\n"
            f"2. Ask one empathetic question at a time.\n"
            f"3. Encourage insights about mood, thoughts, and behaviors.\n"
            f"Begin now with a question relevant to their symptoms."
        )



        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {TOGETHER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "meta-llama/Llama-3-70b-chat-hf",
                "messages": [
                    {"role": "system", "content": "You are a helpful and non-judgmental mental health assistant."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 512
            }
        )

        resp_json = response.json()
        if 'choices' in resp_json and resp_json['choices']:
            return resp_json['choices'][0]['message']['content']
        else:
            return f"LLM Error: {resp_json.get('error', 'No choices returned')}"

    except Exception as e:
        return f"LLM Error: {str(e)}"

# --- Predict and Redirect to Chat ---
@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        required = list(rf_dep.feature_names_in_)
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify(error=f"Missing fields: {', '.join(missing)}"), 400

        ordered = [data[f] for f in required]
        df = pd.DataFrame([ordered], columns=required)

        pred_dep = le_dep.inverse_transform(rf_dep.predict(df))[0]
        pred_anx = le_anx.inverse_transform(rf_anx.predict(df))[0]

        tone = data.get("Tone", "Supportive")
        llama_reply = generate_llama_conversation(data, pred_dep, pred_anx, tone)

        redirect_url = f"/chat?depression={quote(pred_dep)}&anxiety={quote(pred_anx)}&first_question={quote(llama_reply)}&tone={quote(tone)}"

        return jsonify({ "redirect_url": redirect_url })

    except Exception as e:
        return jsonify(error=str(e)), 400

# --- LLaMA Chat Route (POST) ---
@app.route("/api/chat", methods=["POST"])
def llama_chat():
    try:
        payload = request.get_json()
        messages = payload.get("messages", [])

        if not messages:
            return jsonify(error="No messages provided"), 400

        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {TOGETHER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "meta-llama/Llama-3-70b-chat-hf",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 512
            }
        )

        resp_json = response.json()
        if 'choices' in resp_json and resp_json['choices']:
            return jsonify(resp_json['choices'][0]['message'])
        else:
            return jsonify(error=resp_json.get("error", "No choices returned")), 500


    except Exception as e:
        return jsonify(error=f"Server Error: {str(e)}"), 500
    
@app.route("/api/summary", methods=["POST"])
def summary():
    try:
        payload = request.get_json()
        messages = payload.get("messages", [])
        tone = request.args.get("tone", "Supportive")  # get tone from query param

        tone_instructions = {
            "Supportive": "Speak warmly, gently, and compassionately.",
            "Motivational": "Use an uplifting and encouraging tone.",
            "Direct": "Be concise, clear, and to the point with minimal emotion."
        }
        tone_instruction = tone_instructions.get(tone, "Speak warmly and kindly.")

    

        summary_prompt = (
    f"{tone_instruction}\n\n"
    "You are a licensed mental health therapist. Based on the full intake form and this therapeutic chat, please write a final session summary.\n"
    "This user may have experienced emotional overwhelm or even passive suicidal ideation.\n\n"
    "✅ Do NOT write generic advice. Instead:\n"
    "- Reflect their emotional tone and key struggles (as seen in messages)\n"
    "- Confirm if model's depression/anxiety predictions match conversation\n"
    "- Recommend realistic CBT, journaling, and mindfulness exercises with specific examples\n"
    "- Personalize self-care routines for their lifestyle\n"
    "- Offer 2-3 steps in a treatment plan\n"
    "- Finish with a compassionate message of encouragement and suggest professional help if needed\n\n"
    "✅ Format the result in real HTML like this:\n"
    "<h4>Session Summary</h4>\n<p>...</p>\n"
    "<h4>Prediction Alignment</h4>\n<p>...</p>\n"
    "<h4>Therapy Recommendations</h4>\n<ol>...</ol>\n"
    "<h4>Self-Care Plan</h4>\n<ul>...</ul>\n"
    "<h4>Treatment Plan</h4>\n<p>...</p>\n"
    "<h4>Final Encouragement</h4>\n<p>...</p>\n"
    "<h4>Next Steps</h4>\n<ul>...</ul>\n\n"
    "Use <p>, <h4>, <ul>, <ol>, <strong> properly. Do not return Markdown or plain text. Limit to 600 words. Write with compassion and clinical accuracy."
)



        # Add final system + user prompts
        messages.append({"role": "system", "content": f"You are a helpful mental health assistant. {tone_instruction}"})
        messages.append({"role": "user", "content": summary_prompt})

        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {TOGETHER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "meta-llama/Llama-3-70b-chat-hf",
                "messages": messages,
                "temperature": 0.5,
                "max_tokens": 800
            }
        )

        resp_json = response.json()
        if 'choices' in resp_json and resp_json['choices']:
            return jsonify(resp_json['choices'][0]['message'])
        else:
            return jsonify(error=resp_json.get("error", "No choices returned")), 500


    except Exception as e:
        return jsonify(error=f"Summary Error: {str(e)}"), 500


# --- UI Routes ---
@app.route("/")
def home():
    return render_template("mainIndex.html")

@app.route("/get-started")
def get_started():
    return render_template("get-started.html")

@app.route("/predict_form")
def predict_form():
    return render_template("index.html")  # form page

@app.route("/chat")
def chat_page():
    return render_template("chat.html")

@app.route("/submit-register", methods=["POST"])
def submit_register():
    # logic to store data
    return "Thanks for signing up"
@app.route("/login")
def login():
    return render_template("login.html")


# --- Run the App ---
if __name__ == "__main__":
    app.run(debug=True)