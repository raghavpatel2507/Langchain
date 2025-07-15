import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# ------------------- UI Setup ----------------------
st.set_page_config("🎬 YouTube ChatBot", layout="centered")
st.title("🎬 YouTube Video Q&A Chatbot")
st.markdown("Enter a YouTube video URL and ask anything about its content!")

# ------------------ API Input ----------------------
groq_api_key = st.text_input("🔑 Enter your GROQ API key", type="password")
video_url = st.text_input("📺 YouTube Video URL")

# ------------------ Processing Logic ----------------------
def extract_video_id(url):
    if "v=" in url:
        return url.split("v=")[-1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0]
    return url  # if user enters just the ID

def load_transcript(video_id):
    try:
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        return " ".join([chunk["text"] for chunk in transcript_data])
    except TranscriptsDisabled:
        return None

def create_chain_from_transcript(transcript, groq_api_key):
    # Step 1: Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    # Step 2: Embedding + FAISS
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embedding)

    # Step 3: Retriever
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Step 4: LLM + Prompt
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=groq_api_key)

    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        {context}
        Question: {question}
        """,
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Step 5: Chain
    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    main_chain = parallel_chain | prompt | llm | StrOutputParser()
    return main_chain

# -------------------- Chain Execution --------------------
if st.button("🚀 Process Video"):
    if not video_url or not groq_api_key:
        st.warning("Please provide both the video URL and GROQ API key.")
    else:
        with st.spinner("🔍 Fetching transcript and preparing chain..."):
            video_id = extract_video_id(video_url)
            transcript = load_transcript(video_id)

            if not transcript:
                st.error("❌ Transcript not available for this video.")
            else:
                st.session_state.chain = create_chain_from_transcript(transcript, groq_api_key)
                st.success("✅ Ready! You can now ask questions below.")

# ------------------ Ask Questions ----------------------
if "chain" in st.session_state:
    query = st.text_input("💬 Ask a question about the video:")
    if query:
        with st.spinner("🤖 Thinking..."):
            try:
                answer = st.session_state.chain.invoke(query)
                st.markdown(f"**🧠 Answer:** {answer}")
            except Exception as e:
                st.error(f"Error: {e}")
