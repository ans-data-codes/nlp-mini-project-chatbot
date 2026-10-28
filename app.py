"""Prompt Engineering Playground for NLP Experiment No. 9, Group C."""

import re

import streamlit as st
from openai import OpenAI, OpenAIError


# Change this value when you want to try another model supported by Groq.
MODEL_ID = "openai/gpt-oss-20b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
TEMPERATURE = 0.3
LABELS = ("Booking", "Complaint", "Information")


def create_groq_client() -> tuple[OpenAI | None, str | None]:
    """Create a client from Streamlit Secrets, or return a setup message."""
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except (FileNotFoundError, KeyError):
        return None, "Add GROQ_API_KEY to Streamlit Secrets to use the model."

    if not isinstance(api_key, str) or not api_key.strip():
        return None, "GROQ_API_KEY is empty. Add a valid key to Streamlit Secrets."

    return OpenAI(api_key=api_key.strip(), base_url=GROQ_BASE_URL), None


def get_completion(
    client: OpenAI, system_prompt: str, user_prompt: str
) -> tuple[str | None, str | None]:
    """Send one prompt to Groq and return the text or a safe error message."""
    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
        )
    except OpenAIError as error:
        return None, f"Groq request failed ({type(error).__name__}). Check your key, model, or connection."

    answer = response.choices[0].message.content
    if not answer:
        return None, "Groq returned an empty response. Please try again."
    return answer.strip(), None


def show_secret_setup() -> None:
    """Explain how to add the key without asking the user to paste it here."""
    st.error("Add your Groq key to Streamlit Secrets before running this task.")
    st.code('GROQ_API_KEY = "your_groq_api_key"', language="toml")
    st.caption("Do not put the key in app.py or commit your secrets.toml file.")


def run_prompt_comparison(
    client: OpenAI,
    basic_prompt: str,
    engineered_prompt: str,
    user_content: str,
) -> None:
    """Run both versions of a task and show their prompts and responses."""
    basic_system = "You are a helpful assistant."
    engineered_system = (
        "Follow the instructions carefully. Give a clear, accurate answer. "
        "Do not invent facts that are not supported by the input."
    )

    with st.spinner("Comparing the two prompts..."):
        basic_answer, basic_error = get_completion(
            client, basic_system, f"{basic_prompt}\n\n{user_content}"
        )
        if basic_error:
            st.error(basic_error)
            return

        engineered_answer, engineered_error = get_completion(
            client, engineered_system, f"{engineered_prompt}\n\n{user_content}"
        )

    if engineered_error:
        st.error(engineered_error)
        return

    with st.expander("View the prompts used"):
        st.markdown("**Basic prompt**")
        st.code(f"System: {basic_system}\nUser: {basic_prompt}\n\n{user_content}")
        st.markdown("**Engineered prompt**")
        st.code(
            f"System: {engineered_system}\nUser: {engineered_prompt}\n\n{user_content}"
        )

    basic_column, engineered_column = st.columns(2)
    with basic_column:
        st.subheader("Basic prompt")
        st.write(basic_answer)
    with engineered_column:
        st.subheader("Engineered prompt")
        st.write(engineered_answer)


def render_question_answering(client: OpenAI | None) -> None:
    st.subheader("Question answering")
    st.write("Ask a question and compare a direct prompt with clearer instructions.")

    question = st.text_area(
        "Your question",
        placeholder="Why do leaves change color in autumn?",
        height=110,
        key="qa_question",
    )
    if st.button("Compare prompts", type="primary", key="qa_compare"):
        if not question.strip():
            st.warning("Enter a question first.")
        elif client is None:
            show_secret_setup()
        else:
            run_prompt_comparison(
                client,
                "Answer this question:",
                "Answer the question in plain language. Start with the direct answer, "
                "then add a short explanation. If you are unsure, say so.",
                f"Question: {question.strip()}",
            )

    with st.expander("Optional consistency check"):
        st.caption("Enter two paraphrases of the same question and compare the answers.")
        first_column, second_column = st.columns(2)
        with first_column:
            first_question = st.text_input("Paraphrase 1", key="qa_paraphrase_1")
        with second_column:
            second_question = st.text_input("Paraphrase 2", key="qa_paraphrase_2")

        if st.button("Check consistency", key="qa_consistency"):
            if not first_question.strip() or not second_question.strip():
                st.warning("Enter both paraphrased questions.")
            elif client is None:
                show_secret_setup()
            else:
                system_prompt = (
                    "Answer questions clearly and accurately in plain language. "
                    "If you are unsure, say so."
                )
                first_answer, first_error = get_completion(
                    client, system_prompt, first_question.strip()
                )
                second_answer, second_error = get_completion(
                    client, system_prompt, second_question.strip()
                )
                if first_error or second_error:
                    st.error(first_error or second_error)
                else:
                    first_result, second_result = st.columns(2)
                    with first_result:
                        st.markdown("**Response to paraphrase 1**")
                        st.write(first_answer)
                    with second_result:
                        st.markdown("**Response to paraphrase 2**")
                        st.write(second_answer)


def render_summarization(client: OpenAI | None) -> None:
    st.subheader("Summarization")
    st.write("Compare a basic summary request with one that sets length and style.")

    source_text = st.text_area(
        "Text to summarize",
        placeholder="Paste a paragraph here...",
        height=180,
        key="summary_text",
    )
    if st.button("Compare summaries", type="primary", key="summary_compare"):
        if not source_text.strip():
            st.warning("Paste a paragraph first.")
        elif client is None:
            show_secret_setup()
        else:
            run_prompt_comparison(
                client,
                "Summarize this text:",
                "Summarize the text in 3 to 4 sentences. Use plain language, keep the "
                "main facts, and do not add details that are not in the text.",
                f"Text to summarize:\n---\n{source_text.strip()}\n---",
            )


def parse_label(answer: str) -> str | None:
    """Find one of the expected labels in the model response."""
    match = re.search(r"\b(Booking|Complaint|Information)\b", answer, re.IGNORECASE)
    if not match:
        return None
    return next(label for label in LABELS if label.casefold() == match.group(1).casefold())


def render_classification(client: OpenAI | None) -> None:
    st.subheader("Few-shot classification")
    st.write("Classify a customer message using three labeled examples.")

    with st.container(border=True):
        st.markdown(
            "**Examples**\n\n"
            "- \"Please book a table for two at 7 p.m.\" → Booking\n"
            "- \"My order arrived damaged.\" → Complaint\n"
            "- \"What time do you open?\" → Information"
        )

    query = st.text_area(
        "Customer message",
        placeholder="Type a message to classify...",
        height=110,
        key="classification_query",
    )
    if st.button("Classify message", type="primary", key="classify"):
        if not query.strip():
            st.warning("Enter a customer message first.")
        elif client is None:
            show_secret_setup()
        else:
            system_prompt = (
                "Classify customer messages using only these labels: Booking, "
                "Complaint, or Information. Return exactly one label and no other text."
            )
            user_prompt = (
                "Use these examples:\n"
                'Message: "Please book a table for two at 7 p.m."\nLabel: Booking\n\n'
                'Message: "My order arrived damaged."\nLabel: Complaint\n\n'
                'Message: "What time do you open?"\nLabel: Information\n\n'
                f"Classify this message:\n---\n{query.strip()}\n---"
            )
            with st.spinner("Classifying the message..."):
                answer, error = get_completion(client, system_prompt, user_prompt)

            if error:
                st.error(error)
            else:
                predicted_label = parse_label(answer or "")
                if predicted_label:
                    st.success(f"Predicted label: {predicted_label}")
                else:
                    st.warning("The model response did not match one of the three labels.")
                st.markdown("**Model response**")
                st.write(answer)


st.set_page_config(
    page_title="Prompt Engineering Playground",
    page_icon="🧪",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1040px; padding-top: 2.2rem;}
    [data-testid="stSidebar"] {border-right: 1px solid rgba(128, 128, 128, 0.18);}
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Experiment 9 · Group C")
task = st.sidebar.selectbox(
    "Choose a task",
    ("Question Answering", "Summarization", "Few-shot Classification"),
)
st.sidebar.divider()
st.sidebar.caption(f"Groq model: `{MODEL_ID}`")

st.title("Prompt Engineering Playground")
st.write("See how clearer instructions and examples change an LLM's response.")
st.caption("Compare prompts for question answering and summarization, or try few-shot classification.")

client, client_error = create_groq_client()
if client_error:
    st.sidebar.info("Add GROQ_API_KEY in Streamlit Secrets to run a task.")

if task == "Question Answering":
    render_question_answering(client)
elif task == "Summarization":
    render_summarization(client)
else:
    render_classification(client)
