# Prompt Engineering Playground

A small Streamlit app for NLP Experiment No. 9 (Group C). It uses Groq's OpenAI-compatible API to show how prompt instructions and examples affect a model's response.

## Tasks

- **Question answering:** compare a basic prompt with one that asks for a direct, plain-language answer. The optional consistency check compares answers to two paraphrased questions.
- **Summarization:** compare a basic request with one that asks for a 3 to 4 sentence summary in plain language.
- **Few-shot classification:** classify a customer message as `Booking`, `Complaint`, or `Information` using examples.

The app currently uses Groq model `openai/gpt-oss-20b`. The model ID is set near the top of `app.py`; check [Groq's supported models](https://console.groq.com/docs/models) if you want to change it.

## Run locally

1. Install Python 3.10 or newer.
2. Create and activate a virtual environment, then install the dependencies:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

3. Create `.streamlit/secrets.toml` and add your own Groq key:

   ```toml
   GROQ_API_KEY = "your_groq_api_key"
   ```

   Keep this file private. It is listed in `.gitignore` and must never be committed.

4. Start the app:

   ```powershell
   streamlit run app.py
   ```

Create a key in the [Groq Console](https://console.groq.com/keys). Do not paste it into `app.py` or into a chat.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), create an app and select this repository, branch, and `app.py` as the main file.
3. Open the app's settings, find **Secrets**, and add:

   ```toml
   GROQ_API_KEY = "your_groq_api_key"
   ```

4. Save the secret and let Streamlit finish deploying the app.

The key belongs in Streamlit Secrets, not in the repository. The app shows a setup message if `GROQ_API_KEY` is missing.
