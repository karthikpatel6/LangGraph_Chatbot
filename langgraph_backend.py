from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage
from langchain_mistralai import ChatMistralAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3
from dotenv import load_dotenv
load_dotenv()

model = ChatMistralAI(model_name="mistral-small-2506",temperature=1)

class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]

def chat_node(state : ChatState):
     messages = state["messages"]
     response = model.invoke(messages)
     return {"messages" : [response]}

conn = sqlite3.connect(database='chatbot.db',check_same_thread=False)

checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node",chat_node)

graph.add_edge(START,"chat_node")
graph.add_edge("chat_node",END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
     all_threads = set()
     for checkpoint in checkpointer.list(None):
          all_threads.add(checkpointer.config['configurable']['thread_id'])

     return list(all_threads)     