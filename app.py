from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io

# 🔥 NEW: Gemini import
import google.generativeai as genai

# 🔥 ADD YOUR API KEY HERE
genai.configure(api_key="AIzaSyAX1RX6EU3F17h6N4IHLwBiwQMHjxcu0VA")

# Use Gemini model
model = genai.GenerativeModel("gemini-pro")

app = FastAPI()

# Enable CORS (important for Flutter)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔥 NEW: AI explanation function
def generate_ai_explanation(rate1, rate2, group1, group2):
    try:
        prompt = f"""
        Group {group1} has selection rate {rate1}.
        Group {group2} has selection rate {rate2}.

        Explain clearly if there is bias in 2 short sentences.
        Also suggest how to fix it.
        """

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return "AI explanation unavailable"


def multi_attribute_fairness(df, target_col):
    results = {}

    for col in df.columns:
        if col == target_col:
            continue

        if df[col].nunique() > 10:
            continue

        groups = df[col].unique()

        if len(groups) < 2:
            continue

        rates = {}

        for g in groups[:2]:
            subset = df[df[col] == g]

            # ✅ FIXED LOGIC (case-insensitive)
            rate = subset[target_col].astype(str).str.strip().str.lower().isin(
                ["yes", "1", "true"]
            ).mean()

            rates[g] = rate

        g1, g2 = list(rates.keys())
        rate1, rate2 = rates[g1], rates[g2]

        disparate_impact = rate1 / rate2 if rate2 != 0 else 0
        demographic_parity = abs(rate1 - rate2)
        fairness_score = (1 - demographic_parity) * 100

        # 🔥 NEW: Call Gemini
        ai_text = generate_ai_explanation(rate1, rate2, g1, g2)

        results[col] = {
            "groups": [g1, g2],
            "rates": [rate1, rate2],
            "disparate_impact": disparate_impact,
            "fairness_score": fairness_score,
            "ai_explanation": ai_text  # 👈 NEW FIELD
        }

    return results


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    result = multi_attribute_fairness(df, "Selected")

    return result