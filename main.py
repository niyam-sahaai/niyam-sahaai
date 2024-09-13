from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from langchain_openai import OpenAI
from langchain.schema import SystemMessage, HumanMessage
import os
import streamlit as st


st.title("Indian Law Assistant - AI")

os.environ["OPENAI_API_KEY"] ="" # please type your OPENAI API KEY
llm = OpenAI(temperature=0)
embeddings = OpenAIEmbeddings()
vector_store = Chroma(persist_directory="chroma_db_legal_bot_part1", embedding_function=embeddings)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k" : 2})


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    query = prompt
    get_documents = retriever.invoke(query)
    metadata = [doc.metadata for doc in get_documents]
    print(metadata)
    combined_input = (
        "You are an lawyer assistant and you are provided with some documents with legal contents that might contain relevant sections or articles which can help you answer the question  and revalidate the sections carefully: "
        +query
        +"\n\nRelevant Documents: \n"
        + "\n\n".join([doc.page_content for doc in get_documents])
        + "\n\nSource: \n"
        + "\n\n"+", ".join([f"{key}: {value}" for key, value in metadata[0].items()])
        + "\n\n Please provide an answer considering the above documents only and also show the source. If the answer is not found, provide ''' Content Not found''' "
    )



    messages = [
        SystemMessage(content ="You are a helpful assistant who answers based on the combined input alone"),
        HumanMessage(content = combined_input)
    ]

    result = llm.invoke(messages)

    response = f"AI Law Assistant: {result}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
