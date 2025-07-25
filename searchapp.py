import streamlit as st
from langchain_groq import ChatGroq
import os 
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
#from langchain_community.tools.duckduckgo_search.tool import DuckDuckGoSearchRun

from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
from langchain_community.tools import WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
load_dotenv()


arxiv_wrapper=ArxivAPIWrapper(top_k_values=1,doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper=WikipediaAPIWrapper(top_k_values=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=wiki_wrapper)

search=DuckDuckGoSearchRun(name="search")

st.title("langchain - chat with search")


st.sidebar.title("Settings")
api_key=st.sidebar.text_input("enter your Groq API key:",type="password")

if "message" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"hi i am a chatbot who can search the web,how can i help you"}
    ]

if "message" not in st.session_state:
    st.session_state.message = []


for msg in st.session_state.message:
    st.chat_message(msg["role"]).write(msg['content'])

prompt = st.chat_input("Enter your query")
if prompt and api_key:
    st.session_state.message.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm=ChatGroq(groq_api_key=api_key,model_name="Llama3-8b-8192",streaming=True)
    tools=[search,arxiv,wiki]

    search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parser_errors=True)

    with st.chat_message("assistant"):
        st_cd=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        #response=search_agent.run(st.session_state.messages,callbacks=[st_cd])
        response = search_agent.run(prompt, callbacks=[st_cd])
        st.session_state.message.append({"role":"assistant","content":response})
        st.write(response)