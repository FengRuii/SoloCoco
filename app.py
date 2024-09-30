import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import requests
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langgraph.graph import END, StateGraph
# For State Graph 
from typing_extensions import TypedDict
import os

local_llm = 'llama3'
llama3 = ChatOllama(model=local_llm, temperature=0)
llama3_json = ChatOllama(model=local_llm, format='json', temperature=0)

chat_prompt = PromptTemplate(
    template="""
    
    <|begin_of_text|>
    
    <|start_header_id|>system<|end_header_id|> 

    You are a magic 8 ball.  Give user concise and engmatic answers in one sentence. You can have random results to the answers.  Don't always answer yes.  Tailor answers to the user based on their question. Always answer with a reason. 
    
    <|eot_id|>
    
    <|start_header_id|>user<|end_header_id|>
    
    Question: {question} 
    
    <|eot_id|>
    
    <|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question"],
)


chat = chat_prompt | llama3 | StrOutputParser()

# Function to generate a contextual answer using the Llama 2 model
def get_magic_8_ball_answer(question):

    reply = chat.invoke({"question": question})
    return reply

# Define the Gradio interface
with gr.Blocks(css="""
    body {background-color: black;}
    .magic8ball {
        position: relative;
        margin: 50px auto;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle at 30% 30%, #1a1a1a, #000000 70%);
        border-radius: 50%;
        box-shadow: inset -20px -20px 60px rgba(255, 255, 255, 0.1),
                    inset 20px 20px 60px rgba(0, 0, 0, 0.5),
                    0 0 30px rgba(0, 0, 0, 0.5);
        border: 2px solid #333;
    }
    .magic8ball::before {
        content: "";
        position: absolute;
        top: -10px;
        left: -10px;
        width: 320px;
        height: 320px;
        border-radius: 50%;
        background: radial-gradient(circle at 30% 30%, rgba(255, 255, 255, 0.2), transparent 70%);
        z-index: -1;
    }
    .window {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 180px;
        height: 180px;
        background: radial-gradient(circle at center, #003366 0%, #001a33 100%);
        border-radius: 50%;
        border: 5px solid #004080;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 10px;
        box-shadow: inset 0 0 30px rgba(0, 0, 0, 0.7);
    }
    .answer {
        color: white;
        text-align: center;
        font-size: 20px;
        max-width: 160px;
        word-wrap: break-word;
        font-family: 'Arial', sans-serif;
    }
""") as demo:
    gr.Markdown("<h1 style='text-align: center; color: white;'>Magic 8-Ball</h1>")

    state = gr.State("Ask me a question")

    def update_display(question):
        if question.strip() == "":
            answer = "Please ask a question."
        else:
            answer = get_magic_8_ball_answer(question)
        return f"""
        <div class="magic8ball">
            <div class="window">
                <p class="answer">{answer}</p>
            </div>
        </div>
        """

    with gr.Column():
        with gr.Row():
            question_input = gr.Textbox(label="Ask the Magic 8-Ball a question:", placeholder="Type your question here")
            submit_button = gr.Button("Ask")
        output_display = gr.HTML(value=update_display(""))

    submit_button.click(fn=update_display, inputs=question_input, outputs=output_display)

demo.launch()
