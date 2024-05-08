import streamlit as st
import logging
import sys
import time
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Function to load data and create index
@st.cache(allow_output_mutation=True)  # Cache to load data only once
def load_data_and_create_index():
    # Load documents
    documents = SimpleDirectoryReader("/content/Data").load_data()
    
    # Create LlamaCPP instance
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

    # Create LangchainEmbedding instance
    embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name="thenlper/gte-large"))

    # Create ServiceContext
    service_context = ServiceContext.from_defaults(
        chunk_size=256,
        llm=llm,
        embed_model=embed_model
    )

    # Create VectorStoreIndex from documents
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    
    return index


def response_within_domain(text):
    healthcare_keywords = [
    "disease", "symptom", "treatment", "medicine", "health", "medical", "doctor", "hospital", "patient", "care",
    "diabetes", "hypertension", "cancer", "arthritis", "asthma", "allergy", "infection", "injury", "trauma",
    "surgery", "operation", "transplant", "vaccination", "medication", "therapy", "rehabilitation",
    "nurse", "pharmacist", "therapist", "specialist", "surgeon",
    "fitness", "nutrition", "diet", "exercise", "yoga", "meditation",
    "diagnosis", "prognosis", "etiology", "pathology", "pharmacology"
]
    for keyword in healthcare_keywords:
        if keyword in text.lower():
            return True
    
    return False

def greetings_or_OOD(prompt , classifier):
    healthcare_keywords = ['nutrition', 'diet', 'hygiene', 'injury', 'greeting' ,
    'programming language', 'hospital', 'symptoms', 'cancer', 'pain', 'heart dieses' ,'X-ray','MRI','CT scan',
    'temperature', 'physical activity', 'medication', 'therapy', 'surgery', 'diagnosis',
    'exercise', 'treatment',
    'illness', 'disability', 'genetics']
    label_info = classifier(prompt, healthcare_keywords , multi_label=True)
    print(label_info)
    if label_info['scores'][0] > 0.65:
      if label_info['labels'][0] in ['greeting' , 'activity']:
        return "greetings"
      elif label_info['labels'][0] in ['programming language']:
          return "OOD"
      else:
          return "health_query"
    return "OOD"

def model_response(prompt,query_engine , classifier):

    out_of_domain_text = "I'm here to assist you with healthcare-related inquiries, it seems like your question falls outside of my current scope of expertise. If you have any health concerns or questions about wellness, medications, or medical conditions, feel free to ask, and I'll do my best to provide you with accurate information and guidance"
    prompt_domain = greetings_or_OOD(prompt , classifier)
    if prompt_domain=="OOD":
        return out_of_domain_text
    elif prompt_domain=="greetings" : 
        if any(word in prompt.lower() for word in ["hi", "hello", "hey"]):
            return "Hello! How can I assist you today?"
        elif "good morning" in prompt.lower():
            return "Hello, good morning! How can I assist you today?"
        elif "good afternoon" in prompt.lower():
            return "Hello, good afternoon! How can I assist you today?"
        elif "good evening" in prompt.lower():
            return "Hello, good evening! How can I assist you today?"
        elif "good night" in prompt.lower():
            return "Sorry, I can't sleep. I'm here to help you 24 hours!"
        elif "how are you" in prompt.lower():
            return "I'm doing well, thank you for asking! How can I assist you today?"
        elif "what's up" in prompt.lower():
            return "Not much, just here to help. How can I assist you?"
        elif "how can you help" in prompt.lower():
            return "I can help with a variety of topics. Just let me know what you need assistance with!"
        elif "tell me about yourself" in prompt.lower():
            return "I'm a chatbot designed to assist you with your queries. Feel free to ask me anything!"
        elif "who created you" in prompt.lower():
            return "I was created by a team of developers at Bacancy Data prophets company. How can I assist you today?"
        elif "where are you from" in prompt.lower():
            return "I exist in the digital realm, here to help you wherever you are!"
        else:
            return "If you have any health concerns or questions about wellness, medications, or medical conditions, feel free to ask, and I'll do my best to provide you with accurate information and guidance"
    else:
        response = query_engine.query(prompt)
        response_text = str(response)  # Convert response to string

        # Check if the response is within the healthcare domain
        if response_within_domain(response_text):
            return response_text
        else:
            return out_of_domain_text
            
def response_generator(prompt,query_engine , classifier):
    response = model_response(prompt,query_engine , classifier)
    for word in response.split():
        yield word + " "
        time.sleep(0.05)
 
def main(): 
    index = load_data_and_create_index()
    query_engine = index.as_query_engine()
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

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
            response = st.write_stream(response_generator(prompt , query_engine , classifier))
        if response: 
            st.session_state.messages.append({"role": "assistant", "content": response})
            
if __name__ == "__main__": 
    main()
