import json
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Data load karo
with open("intents.json") as f:
    data = json.load(f)

print("✅ intents.json load ho gaya")

# 2. Patterns aur labels extract karo
sentences = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        sentences.append(pattern)
        labels.append(intent["tag"])

print(f"✅ Total patterns: {len(sentences)}")
print(f"✅ Intents: {set(labels)}")

# 3. TF-IDF — text ko numbers mein badlo
# TF-IDF = kitni baar word aaya vs kitna important hai
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences)

print("✅ TF-IDF vectorization done")

# 4. Model train karo
model = LogisticRegression()
model.fit(X, labels)

print("✅ Model train ho gaya!")

# 5. Model aur vectorizer save karo
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ model.pkl aur vectorizer.pkl save ho gaye!")
print("\n🎉 Training complete!")