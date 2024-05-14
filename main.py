# Streamlit app that recieves an url as an user input scrap it feed the data to an LLM and display the identified SDGs
import streamlit as st
from langfuse import Langfuse
from langfuse.openai import OpenAI, AsyncOpenAI
from trafilatura import fetch_url, extract
import json
import os
from dotenv import load_dotenv

load_dotenv()

langfuse = Langfuse(
  secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
  public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
  host=os.getenv("LANGFUSE_HOST")
)

prompt = langfuse.get_prompt("sdg_classification_chat", label="production")

client = OpenAI()

# Fetch and parse url
def fetch_and_parse(url):
    html = fetch_url(url)
    return extract(html)


# Get the data from the user
url = st.text_input("Enter the URL of the company's website")
if st.button("Submit"):
    data = fetch_and_parse(url)
    st.write(data)
    compiled_prompt = prompt.compile(input_text=data)


    response = client.chat.completions.create(
            model="gpt-4o-2024-05-13",
            messages=compiled_prompt,
            temperature=0,
            # response_format={"type": "json_object"},
            langfuse_prompt=prompt,
            metadata= data["_metadata"]
        )

    # get json response
    response_llm = response["choices"][0]["message"]["content"]

    # Find ```JSON ... ``` in the response
    start = response_llm.find("```JSON")
    end = response_llm.find("```", start + 1)
    response_json = json.loads(response_llm[start + 6:end])


    sdgs_list = response_json["labeling_results"]

    # Display the identified SDGs in a carousel of cards
    st.write("The identified SDGs are:")
    for sdg in sdgs_list:
        st.write(sdg)

# run the app with streamlit run main.py