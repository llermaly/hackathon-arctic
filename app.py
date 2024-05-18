import streamlit as st
import replicate
from transformers import AutoTokenizer
from transcripts import get_index_retriever, get_full_transcript
from pytube import YouTube

from dotenv import load_dotenv

load_dotenv()

icons = {"assistant": "./Snowflake_Logomark_blue.svg", "user": "⛷️"}

st.set_page_config(page_title="Snowflake Arctic Youtube Q&A Assistant")

col1, col2 = st.columns([1, 4])

col1.image("./Snowflake_Logomark_blue.svg", width=100)

col2.write("## Snowflake Arctic Youtube Q&A Assistant")


st.write(
    "Hi. I'm Arctic, a new, efficient, intelligent, and truly open language model created by Snowflake AI Research. I can assist you answering questions about your video content. Just enter the URL of the video you want me to analyze, and ask me a question. I'll do my best to provide you with the most accurate answer. Let's get started! 🚀🔍📚 "
)


st.sidebar.title("Snowflake Arctic Youtube Q&A Assistant")

if not "video_url" in st.session_state:
    st.session_state.video_url = None

video_url = st.sidebar.text_input(
    "Video URL", value="https://www.youtube.com/watch?v=OmA9YVkZQWY"
)


def load_video_state():
    st.session_state.video_url = video_url


load_video = st.sidebar.button("Load video", on_click=load_video_state)

if not "summaries" in st.session_state:
    st.session_state.summaries = {}


def generate_artic_summary(transcripts: str):
    prompt_str = "You are an expert summarizer, "
    prompt_str += "you are responsible for summarizing youtube video transcripts. "
    prompt_str += "Transcripts are below.\n"
    prompt_str += "-------------------" + "\n"
    prompt_str += transcripts + "\n"
    prompt_str += "-------------------" + "\n"
    prompt_str += "Summarize the given text in a concise and clear way. "

    for event in replicate.stream(
        "snowflake/snowflake-arctic-instruct",
        input={
            "prompt": prompt_str,
        },
    ):
        yield str(event)


if st.session_state.video_url:
    video = YouTube(st.session_state.video_url)
    st.write(f"### {video.title}")
    st.image(video.thumbnail_url, use_column_width=True)

    if st.session_state.summaries.get(video.video_id):
        st.write(st.session_state.summaries.get(video.video_id))
    else:
        with st.spinner("Generating summary..."):
            transcripts = get_full_transcript(video.video_id)
            summary = generate_artic_summary(transcripts)
            full_summary = st.write_stream(summary)
            st.session_state.summaries[video.video_id] = full_summary


@st.cache_resource(show_spinner=False)
def get_tokenizer():
    """Get a tokenizer to make sure we're not sending too much text
    text to the Model. Eventually we will replace this with ArcticTokenizer
    """
    return AutoTokenizer.from_pretrained("huggyllama/llama-7b")


def get_num_tokens(prompt):
    """Get the number of tokens in a given prompt"""
    tokenizer = get_tokenizer()
    tokens = tokenizer.tokenize(prompt)
    return len(tokens)


# Function for generating Snowflake Arctic response


def generate_arctic_response(message: str):
    with st.spinner("Generating response..."):
        retriever = get_index_retriever(st.session_state.video_url)
        docs = retriever.retrieve(message)

    prompt_str = "You are a Q&A assistant, "
    prompt_str += "you are responsible for answering the query with given context. "
    prompt_str += "Context information is below.\n"
    prompt_str += "-------------------" + "\n"
    for doc in docs:
        prompt_str += doc.get_content("llm") + "\n"
    prompt_str += "-------------------" + "\n"
    prompt_str += "Given the context information not prior knowledge, "
    prompt_str += "answer the query in a concise and clear way. "
    prompt_str += "If the answer is not in the context, return 'I don't know'. \n"
    prompt_str += "Query: " + message + "\n"
    prompt_str += "Answer:"

    if get_num_tokens(prompt_str) >= 3072:
        st.error("Message length too long. Please keep it under 3072 tokens.")

    for event in replicate.stream(
        "snowflake/snowflake-arctic-instruct",
        input={
            "prompt": prompt_str,
        },
    ):
        yield str(event)

    source_doc = docs[0]

    def seg_to_time(seg):
        return f"{int(seg/60)}:{int(seg%60):02d}"

    start_time = seg_to_time(source_doc.metadata["start"])
    end_time = seg_to_time(
        source_doc.metadata["start"] + source_doc.metadata["duration"]
    )

    st.write("Citation source: ")
    st.markdown(f"> {source_doc.get_content()}")
    st.write(f"From {start_time} to {end_time}")
    st.video(
        st.session_state.video_url,
        start_time=docs[0].metadata["start"],
    )


user_message = st.text_input("Enter your question here", key="prompt")
if st.button("Generate response", use_container_width=True) and user_message:
    with st.chat_message("assistant", avatar="./Snowflake_Logomark_blue.svg"):
        response = generate_arctic_response(user_message)
        full_response = st.write_stream(response)
