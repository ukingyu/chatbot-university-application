# College Application Chatbot

This project is a **Retrieval-Augmented Generation (RAG) Chatbot** designed to assist users with college application-related questions. The chatbot leverages advanced natural language processing (NLP) techniques and a robust backend architecture to provide accurate and context-rich responses.

## Features
- **Interactive Front-End**: Built using **Streamlit** and deployed on **Hugging Face Spaces** for seamless user interaction.
- **Backend Architecture**: Engineered with **AWS API Gateway**, **Lambda**, **DynamoDB**, **Pinecone**, and **Bedrock** for scalable and efficient operations.
- **Dynamic Workflows**: Built using **LangGraph**, enabling the LLM to evaluate user input and dynamically query documents when necessary.
- **Context-Rich Responses**: Powered by **Claude 3.5 Haiku** for generating detailed and contextually relevant answers.
- **Document-to-Vector Conversion**: Utilizes **Amazon Titan Text Embeddings V2** to convert documents into vectors for efficient retrieval.
- **Vector Storage**: Integrated with **Pinecone** for high-performance vector storage and retrieval.
- **Chat History Management**: Designed robust **Lambda** logic to manage chat history between **DynamoDB** and **LangGraph**.

## Deployment
The chatbot is deployed on Hugging Face Spaces and can be accessed here:  
[Chatbot for College Applications](https://huggingface.co/spaces/ukingyu/chatbot-help-college-application)

## Technologies Used
- **Front-End**: Streamlit
- **Back-End**: AWS API Gateway, Lambda, DynamoDB, Pinecone, Bedrock
- **Models**: Amazon Titan Text Embeddings V2, Claude 3.5 Haiku
- **Workflow Management**: LangGraph

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

