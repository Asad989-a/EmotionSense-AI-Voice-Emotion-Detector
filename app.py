import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import Streamlit 
import os

# ------------------------------------------------------------
# Load model and tokenizer
# ------------------------------------------------------------
print("Loading microsoft/DialoGPT-medium locally on CPU ...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# ------------------------------------------------------------
# Chatbot conversation logic
# ------------------------------------------------------------
def chat_response(message, history):
    if history is None:
        history = []

    # Convert message history from message objects → plain text
    formatted_history = []
    for msg in history:
        if isinstance(msg, dict):
            if msg["role"] == "user":
                formatted_history.append((msg["content"], ""))
            elif msg["role"] == "assistant" and formatted_history:
                formatted_history[-1] = (formatted_history[-1][0], msg["content"])

    # Encode input
    new_input_ids = tokenizer.encode(message + tokenizer.eos_token, return_tensors='pt')

    # Build chat context
    bot_input_ids = torch.cat([
        tokenizer.encode(m + tokenizer.eos_token, return_tensors='pt')
        for m, _ in formatted_history
    ] + [new_input_ids], dim=-1) if formatted_history else new_input_ids

    # Generate response
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,
        top_p=0.95,
        do_sample=True
    )

    reply = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Append messages in new dictionary format
    new_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply}
    ]

    return new_history, new_history

# ------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------
with gr.Blocks(title="AI Voice Chatbot") as demo:
    gr.Markdown("## 🤖 AI Chatbot (Voice + Text)")
    gr.Markdown("Chat with DialoGPT — type or record your message below.")

    chatbox = gr.Chatbot(label="Conversation", type="messages")

    with gr.Row():
        text_input = gr.Textbox(label="💬 Type your message here...")
        send_btn = gr.Button("Send")

    gr.Markdown("### 🎤 Voice Input")
    user_audio = gr.Audio(sources=["microphone"], type="filepath", label="🎙 Record message")

    clear_btn = gr.Button("🧹 Clear Conversation")

    send_btn.click(chat_response, inputs=[text_input, chatbox], outputs=[chatbox, chatbox])
    text_input.submit(chat_response, inputs=[text_input, chatbox], outputs=[chatbox, chatbox])
    clear_btn.click(lambda: [], None, chatbox, queue=False)

# ------------------------------------------------------------
# Launch app
# ------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
