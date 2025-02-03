import os
import json
import boto3
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Define custom prompt template for question-answering
system_message_template = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise."""

session_store = []

# Set up DynamoDB
dynamodb = boto3.client('dynamodb')
DYNAMODB_TABLE = 'Chatbot-ChatHistory'

# Initialize LLM
llm = ChatBedrock(model_id='anthropic.claude-3-5-haiku-20241022-v1:0',
    model_kwargs={'max_tokens': 300, 'temperature': 0.1, 'top_p': 0.9})

# Set up Embedding
embeddings = BedrockEmbeddings(model_id='amazon.titan-embed-text-v2:0')

# Set up Vector_Store
pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
index = pc.Index(os.environ.get('PINECONE_INDEX_NAME'))
vector_store = PineconeVectorStore(embedding=embeddings, index=index)

# Convert MessageState['messages'] to JSON string
def messages_to_string(messages):
    serialized = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            message_type = "human"
        elif isinstance(msg, AIMessage):
            message_type = "ai"
        elif isinstance(msg, SystemMessage):
            message_type = "system"
        else:
            continue # Ignore all other types
        serialized.append({"type": message_type, "content": msg.content})
    return json.dumps(serialized)

# Convert JSON string to MessageState['messages']
def string_to_messages(string):
    deserialized = json.loads(string)
    messages = []
    for item in deserialized:
        message_type = item["type"]
        content = item["content"]
        if message_type == "human":
            messages.append(HumanMessage(content=content))
        elif message_type == "ai":
            messages.append(AIMessage(content=content))
        elif message_type == "system":
            messages.append(SystemMessage(content=content))
        else:
            raise ValueError(f"Unknown message type: {message_type}")
    return messages

# Load chat history as MessageState['messages'] from DynamoDB
def load_chat_history(session_id):
    messages = []
    try:
        response = dynamodb.get_item(TableName=DYNAMODB_TABLE,
            Key={'session_id': {'S': session_id}})
        if 'Item' in response:
            chat_history_data = response['Item']['chat_history']['S']
            messages = string_to_messages(chat_history_data)
    except Exception as e:
        print(f"Error loading chat history: {e}")
    return messages

# Save chat history of MessageState['messages'] to DynamoDB
def save_chat_history(session_id, messages):
    try:
        chat_history_data = messages_to_string(messages)
        dynamodb.put_item(TableName=DYNAMODB_TABLE,
            Item={'session_id': {'S': session_id},
                'chat_history': {'S': chat_history_data}})
    except Exception as e:
        print(f"Error saving chat history: {e}")

# Create a Tool-Call
@tool(response_format='content_and_artifact')
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = '\n\n'.join(
        (f'Source: {doc.metadata}\n' f'Content: {doc.page_content}')
        for doc in retrieved_docs)
    return serialized, retrieved_docs

# Define Nodes of StateGraph
# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state['messages'])
    return {'messages': [response]} # Append response to state

# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])

# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    # Get the most recent ToolMessages
    recent_tool_messages = []
    for message in reversed(state['messages']):
        if message.type == 'tool':
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_context = '\n\n'.join(doc.content for doc in tool_messages)   
    system_context_message = (f'{system_message_template}\n\n{docs_context}')
    chat_history_message = [message for message in state['messages']
        if message.type in ('human', 'system') 
        or (message.type == 'ai' and not message.tool_calls)]
    prompt = [SystemMessage(system_context_message)] + chat_history_message

    # Generate response
    response = llm.invoke(prompt)
    return {'messages': [response]}

# Set up StateGraph
graph_builder = StateGraph(MessagesState)
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

# graph_builder.set_entry_point('query_or_respond')
graph_builder.add_edge(START, 'query_or_respond') 
graph_builder.add_conditional_edges('query_or_respond', 
    tools_condition, {END: END, 'tools': 'tools'})
graph_builder.add_edge('tools', 'generate')
graph_builder.add_edge('generate', END)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

def lambda_handler(event, context):
    session_id = event['session_id']
    user_input = event['user_input']
    config = {'configurable': {'thread_id': session_id}}

    if session_id not in session_store: # Means it doesn't exist in MessageState as well
        messages_history = load_chat_history(session_id)
        if len(messages_history) > 0:
            graph.update_state(config, {'messages': messages_history})
        session_store.append(session_id)

    response = graph.invoke(
        {'messages': [{'type': 'human', 'content': user_input}]}, 
        config=config)
    save_chat_history(session_id, response['messages'])

    return {
        'statusCode': 200,
        'body': json.dumps(response['messages'][-1].content)
    }
