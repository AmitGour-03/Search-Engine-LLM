import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper  ## In this we'll also add one search engine tool
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun ## It helps us to search anything from the Internet
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler  ## It allows us to communicate with all these kind of tools within themselves
import os
from dotenv import load_dotenv
# load_dotenv()  ## Now, no need to do this, bcz I will add in Streamlit web page

## Here, we have used the in-built tool of Arxiv
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250) ## doc_content_chars_max, we can increase this
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

## Here, we have used the in-built tool of Wikipedia
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250) ## doc_content_chars_max, we can increase this
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

## For searching from the internet
search = DuckDuckGoSearchRun(name="Search")

## Setting for the title:
st.title("🔎 LangChain - Chat with search")
"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

## setting for sidebar
st.sidebar.title("Setting")
api_key = st.sidebar.text_input("Enter your Groq API KEY", type="password")
## pehle yha side bar nhi likha tha

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant", "content":"Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt:=st.chat_input(placeholder="Enter your query"):
    st.session_state.messages.append({"role":"user", "content":prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
    tools = [search, arxiv, wiki]

    search_agent = initialize_agent(tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_error = True)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({'role':'assistant', "content":response})
        st.write(response)

