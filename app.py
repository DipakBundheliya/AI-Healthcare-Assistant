import streamlit as st
import logging
import sys
import time
import logging
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.core.memory import ChatMemoryBuffer
 
MEMORY = None
if "MEMORY" not in st.session_state:
  st.session_state.MEMORY = ChatMemoryBuffer.from_defaults(token_limit=1500)

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="logs.log",
)

# Create a logger
logger = logging.getLogger("streamlit_app")

# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Function to load data and create index
@st.cache(allow_output_mutation=True)  # Cache to load data only once
def load_data_and_create_index():
    # Load documents
    documents = SimpleDirectoryReader("/content/Data").load_data()
    logger.info("Documents loaded successfully")

    # Create LlamaCPP instance
    try:
        llm = LlamaCPP(
            model_url='https://huggingface.co/MaziyarPanahi/BioMistral-7B-GGUF/resolve/main/BioMistral-7B.Q4_K_M.gguf',
            temperature=0.1,
            max_new_tokens=256,
            context_window=3900,
            model_kwargs={"n_gpu_layers": -1},
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            verbose=True,
        )
        logger.info("Model download successfully")
    except Exception as e:
        logger.info("Unexpected error : ",e)
        logger.info("Error occurs while downloading..")

    # Create LangchainEmbedding instance
    embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name="thenlper/gte-large"))
    logger.info("Embedding model download successfully")
    # Create ServiceContext
    service_context = ServiceContext.from_defaults(
        chunk_size=256,
        llm=llm,
        embed_model=embed_model
    )

    # Create VectorStoreIndex from documents
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    logger.info("Vectors created successfully")
    return index
 
def model_response(prompt,chat_engine ):
  # assistant_prompt = "You are an Mediguide - AI healthcare assistant , you have to give answer related to medical, healthcare domain and conversational greetings.If question is outside healthcare and outside conversational greetings say that i do not have knowledge outside healthcare domain ,question is :" + prompt
  response = chat_engine.chat(prompt)
  response_text = str(response)  # Convert response to string
  return response_text 

def response_generator(prompt,chat_engine ):
    response = model_response(prompt,chat_engine)
    logger.info("Get model response successfully")
    for word in response.split():
        yield word + " "
        time.sleep(0.05)
 
def main(): 
    index = load_data_and_create_index()
    # memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
    
    chat_engine =  index.as_chat_engine(
    chat_mode="context",
    memory=st.session_state.MEMORY,
    system_prompt=(
    "You are an Mediguide - AI healthcare assistant , you have to give answer related to medical, healthcare domain and conversational greetings.If question is outside healthcare and outside conversational greetings say that i do not have knowledge outside healthcare domain"),
    )
    print(st.session_state.MEMORY)
    st.title("Mediguide: Your virtual assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "context" not in st.session_state:
        st.session_state.context = []
 
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
         
    if prompt := st.chat_input("Ask your query: "):
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt}) 

        with st.chat_message("assistant"):
            response = st.write_stream(response_generator(prompt , chat_engine ))
        if response: 
            st.session_state.messages.append({"role": "assistant", "content": response})
            print(st.session_state.MEMORY)
if __name__ == "__main__": 
    main()
