# Libraries import
import os 
import streamlit as st
from apikey import apikey

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper


os.environ['OPENAI_API_KEY'] = apikey

# Application Framework
st.title(':clapper: Script Generator')
prompt = st.text_input('Insert the topic of the video here:')

# Creating the title of the video
title_template = PromptTemplate(
    input_variables=['topic'],
    template='write me a youtube video title about {topic}'
)

# Creating the script of the video
script_template = PromptTemplate(
    input_variables=['title','wikipedia_research'],
    template='write me a youtube video script based on this title: {title} and also by leveraging this wikipedia research: {wikipedia_research}'
)

# Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# Create LLM Model
LLM = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=LLM, prompt=title_template, output_key='title', memory=title_memory, verbose=True)
script_chain = LLMChain(llm=LLM, prompt=script_template, output_key='script', memory=script_memory, verbose=True)
#sequential_chain = SequentialChain(chains=[title_chain, script_chain], input_variables=['topic'], output_variables=['title', 'script'], verbose=True)

wiki=WikipediaAPIWrapper()

# Generate a response
if prompt:
    title = title_chain.run(prompt)
    wikipedia_research = wiki.run(prompt)
    script = script_chain.run(title=title, wikipedia_research=wikipedia_research)

    st.write(title)
    st.write(script)

    with st.expander('Title History'):
        st.info(title_memory.buffer)

    with st.expander('Script History'):
        st.info(script_memory.buffer)

    with st.expander('Wikipedia Research'):
        st.info(wikipedia_research)
