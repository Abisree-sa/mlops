"""Simple Gemini chat client for Python.

Requires:
    pip install --upgrade google-generativeai

Authentication options:
    - Set GOOGLE_API_KEY with your API key
    - or configure GOOGLE_APPLICATION_CREDENTIALS to a service account JSON file

Example:
    $Env:GOOGLE_API_KEY = "YOUR_API_KEY"
    python ai_model.py
"""

import os
import google.generativeai as genai


def configure_gemini_api() -> None:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
    else:
        # If no API key is provided, rely on application default credentials.
        # Ensure GOOGLE_APPLICATION_CREDENTIALS is set if using a service account.
        genai.configure()


def chat_with_gemini(messages, model: str = "gemini-pro-1.0") -> str:
    """Send a chat conversation to Gemini and return the model response."""
    response = genai.chat.create(
        model=model,
        messages=messages,
    )

    # The library may return one or more candidates; prefer the last message.
    if hasattr(response, "last") and response.last:
        return response.last.content

    if response.candidates:
        return response.candidates[0].content

    return ""


def build_message_history(user_prompt: str, history=None):
    if history is None:
        history = []

    history.append({"role": "user", "content": user_prompt})
    return history


def main():
    configure_gemini_api()

    print("Gemini chat client\nType a message and press Enter. Type 'exit' to quit.")

    history = []
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        history = build_message_history(user_input, history)
        assistant_reply = chat_with_gemini(history)
        print(f"Gemini: {assistant_reply}\n")

        history.append({"role": "assistant", "content": assistant_reply})


if __name__ == "__main__":
    main()
