from flask import Flask, request, jsonify, render_template
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
import os

app = Flask(__name__)

# Load models and data
embed_model = SentenceTransformer("./all-MiniLM-L6-v2", device="cpu")
index = faiss.read_index("faiss_index.index")
with open("metadata.json") as f:
    metadata = json.load(f)

# Configure Gemini
genai_api_key ="AIzaSyCIhzKAOCeRUL-GX2q0jbJL6-vgxUMPIeM" # Store your key in env
client = genai.Client(api_key=genai_api_key)
model_name = "gemini-2.0-flash"

def ask_gemini(prompt: str, use_web: bool = False) -> str:
    try:
        contents = [
            genai.types.Content(role="user", parts=[genai.types.Part(text=prompt)])
        ]
        tools = [genai.types.Tool(google_search=genai.types.GoogleSearch())] if use_web else []
        config = genai.types.GenerateContentConfig(
            tools=tools,
            response_mime_type="text/plain"
        )

        response_stream = client.models.generate_content_stream(
            model="gemini-1.5-pro",
            contents=contents,
            config=config
        )
        result = "".join([chunk.text for chunk in response_stream if chunk.text])
        return result

    except Exception as e:
        print("❌ Gemini ERROR:", str(e))
        return f"⚠️ Gemini Error: {str(e)}"


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"response": "❌ Please enter a valid question."})

    q_embed = embed_model.encode([query])
    D, I = index.search(q_embed, k=3)
    context = "\n\n".join([metadata[str(i)] for i in I[0]])

    prompt = f"""Use the following information to answer the question.\n\nContext:\n{context}\n\nQuestion:\n{query}"""
    answer = ask_gemini(prompt, use_web=False)
    return jsonify({"response": answer})

@app.route("/web-search", methods=["POST"])
def web_search():
    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"response": "❌ Please enter a valid question."})

    answer = ask_gemini(query, use_web=True)
    return jsonify({"response": answer})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))  # Render uses dynamic PORT, default to 10000
    app.run(host="0.0.0.0", port=port)         # Critical: bind to 0.0.0.0 so it's publicly reachable

