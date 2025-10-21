import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import emoji

# ------------------------------------------------------------
# Load model and tokenizer
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    st.write("⏳ Loading DialoGPT-medium model... please wait.")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    return tokenizer, model

tokenizer, model = load_model()

# ------------------------------------------------------------
# Function to generate chatbot reply
# ------------------------------------------------------------
def generate_reply(user_input, chat_history_ids=None):
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    bot_input_ids = (
        torch.cat([chat_history_ids, new_input_ids], dim=-1)
        if chat_history_ids is not None
        else new_input_ids
    )

    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    reply = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return reply, chat_history_ids

# ------------------------------------------------------------
# Streamlit App UI
# ------------------------------------------------------------
st.set_page_config(page_title="🤖 AI Voice Chatbot", page_icon="🎤", layout="centered")

st.title("🎙 AI Voice + Text Chatbot")
st.markdown("Chat with **Microsoft DialoGPT** – Type your message below.")

if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None
if "past_messages" not in st.session_state:
    st.session_state.past_messages = []

user_input = st.text_input("💬 Type your message here:")

if st.button("Send"):
    if user_input.strip():
        reply, chat_ids = generate_reply(user_input, st.session_state.chat_history_ids)
        st.session_state.chat_history_ids = chat_ids
        st.session_state.past_messages.append(("You", user_input))
        st.session_state.past_messages.append(("Bot", reply))
    else:
        st.warning("Please type something before sending.")

# ------------------------------------------------------------
# Chat Display
# ------------------------------------------------------------
st.markdown("### 💬 Conversation")
for sender, msg in st.session_state.past_messages:
    if sender == "You":
        st.markdown(f"🧑 **You:** {msg}")
    else:
        st.markdown(f"🤖 **Bot:** {msg}")

# ------------------------------------------------------------
# Reset Chat
# ------------------------------------------------------------
if st.button("🧹 Clear Conversation"):
    st.session_state.past_messages = []
    st.session_state.chat_history_ids = None
    st.success("Conversation cleared!")
