from flask import Flask, request, Response, render_template, session
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
import threading
import queue
import time

app = Flask(__name__)
app.secret_key = "your-secret-key"

llm = None

def init_model():
    global llm
    try:
        llm = Ollama(model="llama3.2")
        print("✅ Model initialized successfully.")
    except Exception as e:
        print(f"❌ Error initializing model: {e}")

# Initialize the model in a separate thread
threading.Thread(target=init_model, daemon=True).start()

# Helper function to get or create session memory
def get_memory():
    if 'memory' not in session:
        session['memory'] = ConversationBufferMemory()  # Create new memory for this user session
    return session['memory']

def generate_response_stream(prompt, user_input, memory):
    chunk_queue = queue.Queue()
    full_response = []

    def fetch_chunks():
        if not llm:
            return
        for chunk in llm.stream(prompt):
            if 'stop_event' in session and session['stop_event'].is_set():
                break
            chunk_queue.put(chunk)
            full_response.append(chunk)

    thread = threading.Thread(target=fetch_chunks)
    thread.start()

    def stream():
        while thread.is_alive() or not chunk_queue.empty():
            if not chunk_queue.empty():
                chunk = chunk_queue.get()
                yield chunk
            if 'stop_event' in session and session['stop_event'].is_set():
                break

        # Save the full response in memory after generation completes
        memory.save_context({"input": user_input}, {"output": "".join(full_response)})

    return Response(stream(), content_type="text/html", status=200)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return Response("⚠️ Please provide a valid input.", content_type="text/plain")

    # Get the unique memory for the current user session
    memory = get_memory()

    stop_event = threading.Event()
    session['stop_event'] = stop_event  # Save the stop_event in the session for controlling generation

    context = memory.load_memory_variables({})
    system_prompt = "You are AstroGPT, a helpful assistant. Do not introduce yourself if already done."
    history_prompt = f"{system_prompt}\n{context.get('history', '')}\nUser: {user_input}\nAstroGPT:"

    return generate_response_stream(history_prompt, user_input, memory)

@app.route("/stop", methods=["POST"])
def stop():
    # Trigger stop event to halt generation
    if 'stop_event' in session:
        session['stop_event'].set()
    return Response("⏹️ Text generation stopped.", content_type="text/plain")

@app.route("/reset", methods=["POST"])
def reset():
    # Reset the memory for this session
    memory = get_memory()
    memory.clear()
    return Response("♻️ Conversation history cleared.", content_type="text/plain")

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=False)
