from flask import Flask, request, jsonify
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os

app = Flask(__name__)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Index creator
@app.route("/create_index", methods=["POST"])
def create_index():
    data = request.json
    patient_id = data["patient_id"]
    text = data["text"]

    os.makedirs("patients", exist_ok=True)
    os.makedirs("indexes", exist_ok=True)

    with open(f"./patients/{patient_id}.txt", "w", encoding="utf-8") as f:
        f.write(text)

    chunks = text.split("\n")
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    faiss.write_index(index, f"./indexes/{patient_id}.index")

    with open(f"./indexes/{patient_id}_texts.json", "w") as f:
        json.dump(chunks, f)

    return jsonify({"status": " Index created for " + patient_id})


# LLM question answering
@app.route("/query_patient", methods=["POST"])
def query_patient():
    data = request.json
    patient_id = data["patient_id"]
    user_question = data["question"]

    index = faiss.read_index(f"./indexes/{patient_id}.index")
    with open(f"./indexes/{patient_id}_texts.json", "r") as f:
        texts = json.load(f)

    query_vector = model.encode([user_question])
    distances, indices = index.search(np.array(query_vector), k=3)
    context = "\n".join([texts[i] for i in indices[0]])

    prompt = f"""You are a helpful assistant.
Based on this patient information:
{context}

Answer the question: {user_question}
"""

    #Call local LLaMA 3.1
    import requests
    llama_response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.1",
            "prompt": prompt,
            "stream": False
        }
    ).json()["response"]

    return jsonify({"answer": llama_response})


if __name__ == "__main__":
    app.run(port=5000)
