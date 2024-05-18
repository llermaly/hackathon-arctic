from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
from llama_index.core import Document
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import ServiceContext
from llama_index.llms.replicate import Replicate


def group_transcripts_by_character_count(transcripts, char_count):
    grouped = []
    current_group = {
        "text": "",
        "start": 0,
        "duration": 0,
    }

    for index, transcript in enumerate(transcripts):
        if len(current_group["text"]) == 0:
            current_group["start"] = transcript["start"]

        if len(current_group["text"]) + len(transcript["text"]) <= char_count:
            current_group["text"] += transcript["text"] + " "
            current_group["duration"] = (
                transcript["start"] + transcript["duration"] - current_group["start"]
            )
        else:
            grouped.append(current_group)
            current_group = {
                "text": transcript["text"] + " ",
                "start": transcript["start"],
                "duration": transcript["duration"],
            }

        if index == len(transcripts) - 1:
            grouped.append(current_group)

    return grouped


def get_full_transcript(video_id: str):
    list_transcript = YouTubeTranscriptApi.list_transcripts(video_id)

    manually_list = list_transcript._manually_created_transcripts
    lang = "en"

    for key in manually_list:
        if key:
            lang = key
            break

    transcript_lines = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])

    full_transcript = ""

    for transcript in transcript_lines:
        full_transcript += transcript["text"] + " "

    return full_transcript


def get_yt_documents(video_url: str):
    yt = YouTube(video_url)

    list_transcript = YouTubeTranscriptApi.list_transcripts(yt.video_id)

    manually_list = list_transcript._manually_created_transcripts
    lang = "en"

    for key in manually_list:
        if key:
            lang = key
            break

    transcript_lines = YouTubeTranscriptApi.get_transcript(
        yt.video_id, languages=[lang]
    )

    grouped_transcripts = group_transcripts_by_character_count(transcript_lines, 400)

    documents = []

    shared_metadata = {
        "title": yt.title or "Unknown",
        "description": yt.description or "Unknown",
        "view_count": yt.views or 0,
        "thumbnail_url": yt.thumbnail_url or "Unknown",
        "publish_date": (
            yt.publish_date.strftime("%Y-%m-%d %H:%M:%S")
            if yt.publish_date
            else "Unknown"
        ),
        "length": yt.length or 0,
        "author": yt.author or "Unknown",
        "source": yt.video_id,
    }

    for transcript in grouped_transcripts:
        doc = Document(
            text=transcript["text"],
            metadata={
                "start": transcript["start"],
                "duration": transcript["duration"],
                **shared_metadata,
            },
        )
        documents.append(doc)

    return documents


def get_index_retriever(video_url: str) -> BaseRetriever:
    documents = get_yt_documents(video_url)

    embed_model = HuggingFaceEmbedding(model_name="Snowflake/snowflake-arctic-embed-l")

    llm = Replicate(model="Snowflake/snowflake-arctic-instruct-vllm")

    service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)

    index = VectorStoreIndex.from_documents(documents, service_context=service_context)

    return index.as_retriever(similarity_top_k=1)
