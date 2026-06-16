import json
import pickle
import random
import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. Environment variables load karo (.env file se)
load_dotenv()

# 2. OpenRouter client banao
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# 3. Saved ML model aur vectorizer load karo
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# 4. Intents load karo
with open("intents.json") as f:
    data = json.load(f)

print("✅ Chatbot ready! (ML + AI hybrid)")

# 5. AI se response lene wala function
def get_ai_response(user_input):
    try:
        response = client.chat.completions.create(
            model="openrouter/free",
            messages=[
                {"role": "user", "content": user_input}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI error aaya: {str(e)}"

# 6. Main response function — hybrid logic
def get_response(user_input):
    # ML model se predict karo
    X = vectorizer.transform([user_input])
    intent = model.predict(X)[0]
    confidence = max(model.predict_proba(X)[0])

    # Agar ML bohot confident hai → ML response do (fast)
    if confidence >= 0.6:
        for i in data["intents"]:
            if i["tag"] == intent:
                return random.choice(i["responses"])

    # Agar ML confident nahi hai, ya match nahi mila → AI se poocho
    return get_ai_response(user_input)

# 7. Quick test
if __name__ == "__main__":
    print("Test kar raha hun...\n")
    test_inputs = ["Hello", "What is machine learning?", "Bye", "Explain neural networks in simple words"]

    for text in test_inputs:
        response = get_response(text)
        print(f"User: {text}")
        print(f"Bot:  {response}\n")