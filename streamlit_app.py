import streamlit as st
from openai import OpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import UnstructuredFileIOLoader
from langchain.embeddings import HuggingFaceEmbeddings
from typing import Optional, List, Mapping, Any



# Show title and description.
st.title("💬 Chatbot")
st.write(
    "This is a simple chatbot that uses OpenAI's GPT-3.5 model to generate responses. "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
    "You can also learn how to build this app step by step by [following our tutorial](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)."
)

class CustomGraniteLLM(LLM):
    url: str  # URL of the Granite model endpoint
    headers: Optional[Mapping[str, str]] = None  # Optional headers for authentication
    max_tokens: int = 500  # Max tokens to generate
    temperature: float = 0.7  # Sampling temperature

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Prepare the payload for the API request
        payload = {
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        # Make the API request
        response = requests.post(self.url, json=payload, headers=self.headers)
        response.raise_for_status()
        # Extract the generated text from the response
        return response.json()["generated_text"]

    @property
    def _llm_type(self) -> str:
        return "custom_granite_llm"

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:

    #Load and split documents
    loader = WebBaseLoader("https://www.burlington.ca/Modules/News/en/Snow")
    webdocs = loader.load()



    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(webdocs)

    # Create vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(texts, embeddings)

    # Build QA chain
    llm = ChatOpenAI(model="gpt-3.5-turbo",api_key=openai_api_key)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_db.as_retriever(),
        chain_type="stuff"
    )

    #response = qa.run("What is RAG?")


    # Create a session state variable to store the chat messages. This ensures that the
    # messages persist across reruns.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display the existing chat messages via `st.chat_message`.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Create a chat input field to allow the user to enter a message. This will display
    # automatically at the bottom of the page.
    if prompt := st.chat_input("What is up?"):

        # Store and display the current prompt.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate a response using the OpenAI API.
        #stream = client.chat.completions.create(
        #    model="gpt-4o-mini",
        #    messages=[
        #        {"role": m["role"], "content": m["content"]}
        #        for m in st.session_state.messages
        #    ],
        #    stream=True,
        #)

        # Stream the response to the chat using `st.write_stream`, then store it in 
        # session state.
        with st.chat_message("assistant"):
            response = st.write(qa.run(prompt))
        st.session_state.messages.append({"role": "assistant", "content": response})
