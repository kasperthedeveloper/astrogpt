from flask import Flask, request, Response, render_template, session
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
import threading
import queue

app = Flask(__name__)
app.secret_key = "your-secret-key"

llm = None
user_sessions = {}  # Store memory and stop events per user session

def init_model():
    """Initialize the AI model."""
    global llm
    try:
        llm = Ollama(model="llama3.2")
        print("✅ Model initialized successfully.")
    except Exception as e:
        print(f"❌ Error initializing model: {e}")

# Start model initialization in a separate thread
threading.Thread(target=init_model, daemon=True).start()

def get_user_memory(session_id):
    """Retrieve or create a ConversationBufferMemory for the session."""
    if session_id not in user_sessions:
        user_sessions[session_id] = {
            "memory": ConversationBufferMemory(),
            "stop_event": threading.Event(),
        }
    return user_sessions[session_id]["memory"]

def get_stop_event(session_id):
    """Retrieve the stop event for the session."""
    return user_sessions.get(session_id, {}).get("stop_event")

def generate_response_stream(prompt, user_input, memory, stop_event):
    """Generates a response stream while ensuring session safety."""
    chunk_queue = queue.Queue()
    full_response = []

    def fetch_chunks():
        if not llm:
            return
        for chunk in llm.stream(prompt):
            if stop_event.is_set():
                break
            chunk_queue.put(chunk)
            full_response.append(chunk)

    thread = threading.Thread(target=fetch_chunks)
    thread.start()

    def stream():
        while thread.is_alive() or not chunk_queue.empty():
            if not chunk_queue.empty():
                yield chunk_queue.get()
            if stop_event.is_set():
                break

        # Save the full response in memory after generation completes
        memory.save_context({"input": user_input}, {"output": "".join(full_response)})

    return Response(stream(), content_type="text/plain")

@app.route("/chat", methods=["POST"])
def chat():
    """Handles user messages and generates AI responses."""
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return Response("⚠️ Please provide a valid input.", content_type="text/plain")

    if "session_id" not in session:
        session["session_id"] = str(id(session))  # Unique identifier for the session

    session_id = session["session_id"]
    memory = get_user_memory(session_id)
    stop_event = get_stop_event(session_id)
    
    if stop_event:
        stop_event.clear()  # Reset stop event before generating response

    context = memory.load_memory_variables({})
    system_prompt = "You are AstroGPT, a helpful assistant. Do not introduce yourself if already done."
    history_prompt = f"{system_prompt}\n{context.get('history', '')}\nUser: {user_input}\nAstroGPT:"

    return generate_response_stream(history_prompt, user_input, memory, stop_event)

@app.route("/stop", methods=["POST"])
def stop():
    """Stops the ongoing AI response generation."""
    session_id = session.get("session_id")
    stop_event = get_stop_event(session_id)
    if stop_event:
        stop_event.set()
    return Response("⏹️ Text generation stopped.", content_type="text/plain")

@app.route("/reset", methods=["POST"])
def reset():
    """Clears the conversation memory for this session."""
    session_id = session.get("session_id")
    if session_id in user_sessions:
        user_sessions[session_id]["memory"].clear()
    return Response("♻️ Conversation history cleared.", content_type="text/plain")

@app.route("/")
def home():
    """Renders the main chat page."""
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=False)
