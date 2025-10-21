import streamlit as st
import soundfile as sf
import numpy as np
import tempfile
import os
import torch
import torchaudio
from transformers import pipeline

# -----------------------------
# 🎧 Load Model
# -----------------------------
st.set_page_config(page_title="🎤 Voice Emotion Detection", page_icon="🎶", layout="centered")

@st.cache_resource
def load_model():
    return pipeline("audio-classification", model="superb/hubert-large-superb-er")
model = load_model()

# -----------------------------
# 🎨 Title + UI
# -----------------------------
st.title("🎙️ Voice Emotion Detector with Emoji")
st.markdown("Upload or record your voice to detect its **emotion** 🎭")

# -----------------------------
# 🎤 Audio Input
# -----------------------------
st.sidebar.header("🎚️ Audio Controls")
record_option = st.sidebar.radio("Choose Input Method:", ["🎙️ Record Voice", "📁 Upload File"])

audio_bytes = None
if record_option == "📁 Upload File":
    uploaded_file = st.file_uploader("Upload an audio file (wav/mp3)", type=["wav", "mp3"])
    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
else:
    st.info("⚠️ Streamlit doesn’t yet support native microphone recording directly. Use upload for now or integrate Streamlit-webrtc.")
    # Optionally you can add `streamlit-webrtc` for live mic input.

# -----------------------------
# 🧠 Process and Predict
# -----------------------------
if audio_bytes:
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    st.audio(audio_bytes, format="audio/wav")

    # Run model
    result = model(tmp_path)[0]
    emotion = result["label"].capitalize()
    score = result["score"]

    emoji_map = {
        "angry": "😡",
        "happy": "😄",
        "sad": "😢",
        "neutral": "😐",
        "surprise": "😲",
        "fear": "😨",
        "disgust": "🤢"
    }

    emoji = emoji_map.get(emotion.lower(), "🎵")

    st.markdown(f"### {emoji} Emotion: **{emotion}** ({score:.2f})")

    os.remove(tmp_path)

# -----------------------------
# 🗣️ Chat / Conversation Log
# -----------------------------
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

user_input = st.text_input("💬 Type your message:")

if user_input:
    st.session_state.chat_log.append({"user": user_input, "bot": f"I can sense your mood might be {emotion.lower()} {emoji}!"})

if st.session_state.chat_log:
    st.markdown("### 🧾 Conversation:")
    for entry in st.session_state.chat_log:
        st.markdown(f"👤 **You:** {entry['user']}")
        st.markdown(f"🤖 **Bot:** {entry['bot']}")
        st.markdown("---")








# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import gradio as gr
# import os

# # ------------------------------------------------------------
# # Load model and tokenizer
# # ------------------------------------------------------------
# print("Loading microsoft/DialoGPT-medium locally on CPU ...")
# tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
# model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# # ------------------------------------------------------------
# # Chatbot conversation logic
# # ------------------------------------------------------------
# def chat_response(message, history):
#     if history is None:
#         history = []

#     # Convert message history from message objects → plain text
#     formatted_history = []
#     for msg in history:
#         if isinstance(msg, dict):
#             if msg["role"] == "user":
#                 formatted_history.append((msg["content"], ""))
#             elif msg["role"] == "assistant" and formatted_history:
#                 formatted_history[-1] = (formatted_history[-1][0], msg["content"])

#     # Encode input
#     new_input_ids = tokenizer.encode(message + tokenizer.eos_token, return_tensors='pt')

#     # Build chat context
#     bot_input_ids = torch.cat([
#         tokenizer.encode(m + tokenizer.eos_token, return_tensors='pt')
#         for m, _ in formatted_history
#     ] + [new_input_ids], dim=-1) if formatted_history else new_input_ids

#     # Generate response
#     chat_history_ids = model.generate(
#         bot_input_ids,
#         max_length=1000,
#         pad_token_id=tokenizer.eos_token_id,
#         temperature=0.7,
#         top_p=0.95,
#         do_sample=True
#     )

#     reply = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

#     # Append messages in new dictionary format
#     new_history = history + [
#         {"role": "user", "content": message},
#         {"role": "assistant", "content": reply}
#     ]

#     return new_history, new_history

# # ------------------------------------------------------------
# # Gradio UI
# # ------------------------------------------------------------
# with gr.Blocks(title="AI Voice Chatbot") as demo:
#     gr.Markdown("## 🤖 AI Chatbot (Voice + Text)")
#     gr.Markdown("Chat with DialoGPT — type or record your message below.")

#     chatbox = gr.Chatbot(label="Conversation", type="messages")

#     with gr.Row():
#         text_input = gr.Textbox(label="💬 Type your message here...")
#         send_btn = gr.Button("Send")

#     gr.Markdown("### 🎤 Voice Input")
#     user_audio = gr.Audio(sources=["microphone"], type="filepath", label="🎙 Record message")

#     clear_btn = gr.Button("🧹 Clear Conversation")

#     send_btn.click(chat_response, inputs=[text_input, chatbox], outputs=[chatbox, chatbox])
#     text_input.submit(chat_response, inputs=[text_input, chatbox], outputs=[chatbox, chatbox])
#     clear_btn.click(lambda: [], None, chatbox, queue=False)

# # ------------------------------------------------------------
# # Launch app
# # ------------------------------------------------------------
# if __name__ == "__main__":
#     demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
