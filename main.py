from flask import Flask, request, jsonify, render_template, session
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
import os

app = Flask(__name__)
app.secret_key = "1ef681bcd16d1a2b79b3ecbb30665c4b68eabd41335ce38eeeb061594db3f1fe"  # Needed for session management

# Load models and data
embed_model = SentenceTransformer("./all-MiniLM-L6-v2", device="cpu")
index = faiss.read_index("faiss_index.index")
with open("metadata.json") as f:
    metadata = json.load(f)

# Configure Gemini
genai_api_key = "AIzaSyB4Big5yLZIjH1pehEyJzBuJ9FvkLm_P_Q"  # Replace with your key
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
            model=model_name,
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
    session.clear()
    return render_template("index.html")

@app.route("/reset", methods=["POST"])
def reset():
    session.clear()
    return jsonify({"message": "Session cleared"})

@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    user_input = data.get("query", "").strip()
    if not user_input:
        return jsonify({"response": "❌ Please enter a valid question."})

    conversation = session.get("conversation", [])
    conversation.append({"role": "user", "content": user_input})

    # Internal RAG search
    q_embed = embed_model.encode([user_input])
    D, I = index.search(q_embed, k=3)
    context = "\n\n".join([metadata[str(i)] for i in I[0]])

    # Web Search
    web_search_answer = ask_gemini(user_input, use_web=True)

    # Now prepare the final diagnostic prompt
    if "problem_identified" not in session:
            prompt = (
        "You are a professional maintenance assistant of SY35U EXCAVATOR.\n\n"
        "Use the internal knowledge and web information to understand the problem quickly.\n\n"
        "If necessary, ask **only ONE** very short question to clarify. "
        "Otherwise, proceed to provide a **complete, detailed maintenance solution** without delay.\n\n"
        "When giving the final answer, use **Markdown format**. Bold all section headings.\n\n"
        "Provide the following sections, and **bold** each title:\n"
        "- **Identified Issue**\n"
        "- **Expert Require**\n"
        "- **Required Tools**\n"
        "- **Precautions**\n"
        "- **Step-by-Step Actions**\n\n"
        "📚 Internal Knowledge:\n"
        f"{context}\n\n"
        "🌐 Web Search Info:\n"
        f"{web_search_answer}\n\n"
        "🧑 Conversation History:\n"
        + "\n".join([f"{turn['role'].capitalize()}: {turn['content']}" for turn in conversation])
    )

    else:
        prompt = (
            "Based on the conversation, internal data, and web info, provide a very detailed maintenance solution.\n\n"
            "Include:\n- Identified Issue\n- Required Tools\n- Precautions\n- Step-by-Step Actions.\n\n"
            f"📚 Internal Knowledge:\n{context}\n\n"
            f"🌐 Web Search Info:\n{web_search_answer}\n\n"
            "🧑 Conversation History:\n" + "\n".join([f"{turn['role'].capitalize()}: {turn['content']}" for turn in conversation])
        )


    final_answer = ask_gemini(prompt, use_web=False)

    # If solution is found, mark it
    if any(kw in final_answer.lower() for kw in ["steps:", "precautions", "required tools", "solution"]):
        session["problem_identified"] = True

    session["conversation"] = conversation
    return jsonify({"response": final_answer})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
