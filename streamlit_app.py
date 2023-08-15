import streamlit as st
from langchain import PromptTemplate
from langchain.llms import Replicate
import os 

st.set_page_config(page_title="🦜🔗 Blog Outline Generator App")
st.title('🦜🔗 Blog Outline Generator App')
jina_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

def generate_response(topic):
  llm = Replicate(
    model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
    input={"temperature": 0.75, "max_length": 500, "top_p": 1},
)
  # Prompt
  template = 'As an experienced data scientist and technical writer, generate an outline for a blog about {topic}.'
  prompt = PromptTemplate(input_variables=['topic'], template=template)
  prompt_query = prompt.format(topic=topic)
  # Run LLM model and print out response
  response = llm(prompt_query)
  return st.info(response)

with st.form('myform'):
  topic_text = st.text_input('Enter keyword:', '')
  submitted = st.form_submit_button('Submit')
  if submitted :
    os.environ["REPLICATE_API_TOKEN"] = jina_api_key
    generate_response(topic_text)