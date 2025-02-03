import os
import streamlit as st
import requests
import uuid

API_URL = os.environ.get("AWS_INVOKE_URL")
st.title('Hi, This is AppliBot')

intro_message = """Need help navigating the maze of college applications? 
    Deadlines, recommendation letters, essays—I’ve got you covered. 
    Let’s tackle it together, one question at a time!"""

example_message = """For example: 
When is the deadline for MIT undergraduate application? 
How many recommendation letters are required?"""

# Init session_id
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Init chat_history and display
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [
        {'role': 'assistant', 'text': intro_message}, 
        {'role': 'assistant', 'text': example_message}]
for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['text'])

def clean_message(message):
    # Remove quote ""
    message = message[1:-1]

    # Take care of \n\n
    message = message.replace("\\n", "\n") # \n(\\n) -> actual newline
    message = message.replace("\n\n", "\n") 

    return message.strip()

# Input user message
input_text = st.chat_input('Chat here')
if input_text:
    with st.chat_message('user'):
        st.markdown(input_text)
    st.session_state.chat_history.append({
        'role': 'user', 'text': input_text})
    try:
        response_raw = requests.post(API_URL,
            params={'session_id': st.session_state.session_id, 
                'user_input': input_text}
        )
        response_data = response_raw.json()
        if response_data.get("statusCode") == 200:
            chat_response = response_data.get('body')
            chat_response = clean_message(chat_response)
        else:
            chat_response = "Sorry, some errors. Can you ask again?"

        with st.chat_message('assistant'):
            st.markdown(chat_response)
        st.session_state.chat_history.append(
            {'role': 'assistant', 'text': chat_response})
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with the backend: {e}")
