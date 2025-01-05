from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# Load Hugging Face model and tokenizer
model_name = "microsoft/DialoGPT-medium"  # Example model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create a conversational pipeline
hf_pipeline = pipeline("conversational", model=model, tokenizer=tokenizer)

# Wrap the pipeline in LangChain's LLM class
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Create a simple conversation chain
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)

# Single-turn conversation example
user_input = "Hello, how are you?"
response = conversation.run(input=user_input)
print(f"Chatbot: {response}")
