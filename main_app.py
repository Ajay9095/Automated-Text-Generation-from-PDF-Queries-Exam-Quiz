import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import time
import random
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.environ["GROQ_API_KEY"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

huggingface_api_key = os.environ["HUGGINGFACEHUB_API_TOKEN"]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

if "area" not in st.session_state:
    st.session_state.area = ""

if "exam_questions" not in st.session_state:
    st.session_state.exam_questions = {"mcq": [], "fill_in_blank": []}

st.sidebar.title("Navigation")
option = st.sidebar.selectbox("Select a page:", ["Home", "Exam"])    

st.sidebar.title("                                        ")
st.sidebar.title("                                        ")
st.sidebar.title("                                        ")

st.sidebar.subheader("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings()

        with open("temp_uploaded_file.pdf", "wb") as f:  
            f.write(uploaded_file.getbuffer())

        st.session_state.loader = PyPDFLoader("temp_uploaded_file.pdf")
        st.session_state.docs = st.session_state.loader.load()

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.documents, st.session_state.embeddings)

##### groq models #####
llm = ChatGroq(groq_api_key=groq_api_key, model="llama3-8b-8192")

prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

document_chain = create_stuff_documents_chain(llm, prompt_template)

if option == "Home":
    st.title("Automated Text Generation from PDF Queries")

    st.markdown("""
        <style>
        .chat-container {
            height: 350px;
            overflow-y: auto;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 10px;
            background-color: #2E2E2E; 
            color: #fff; 
        }
                
        .big-question {
            font-size: 18px;
            font-weight: bold;
            color: #fff; 
        }
                
        .chat-container hr {
            border: 0.5px solid #ccc;
        }
                
        .stButton button {
            background-color: red;
            color: white;
            width: 100%;
            height: 50px;
            border-radius: 10px;
        }
        </style>
        """, unsafe_allow_html=True)

    # Display chat history in a scrollable container
    if st.session_state.chat_history:
        chat_display = ""
        for chat in st.session_state.chat_history:
            chat_display += f"<div class='big-question'>Question :- {chat['question']}</div>\n"
            chat_display += f"<div>Answer :- \n\n{chat['answer']}</div>\n"
            chat_display += "<hr>\n"
        
        st.markdown(f"<div class='chat-container'>{chat_display}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='chat-container'>No chat history yet.</div>", unsafe_allow_html=True)

    def update_chat():
        input_text = st.session_state.input_text
        if input_text:
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            #start_time = time.time()
            response = retrieval_chain.invoke({"input": input_text})
            #end_time = time.time()

            st.session_state.chat_history.append({"question": input_text, "answer": response["answer"]})

            # total_time = end_time - start_time
            # st.write(f"Response time: {total_time:.2f} seconds")

            # Clear the input text
            st.session_state.input_text = ""

    with st.form(key='chat_form', clear_on_submit=True):
        st.text_input("Input your question", key="input_text")
        submit = st.form_submit_button(label="Send", on_click=update_chat)

elif option == "Exam":
    st.title("Exam Question Generation")

    num_questions = st.selectbox("Select number of questions to generate:", [5, 10, 15, 20], index=0)

    question_type = st.selectbox("Select the type of questions to answer:", ["All Questions", "Multiple Choice Questions", "Fill-in-the-Blank Only"], index=0)

    if uploaded_file is not None:

        if 'questions_generated' in st.session_state and st.session_state.questions_generated != num_questions:
            st.session_state.exam_questions = {"mcq": [], "fill_in_blank": [], "mcq_answers": [], "fill_in_blank_answers": []}
            st.session_state.questions_generated = num_questions

        mcq_prompt = f"""
        Generate {num_questions} multiple-choice questions based on the following context.
        Each question should have 4 options. Format the questions as "Q1. <Question>", followed by options "A) <Option 1>", "B) <Option 2>", "C) <Option 3>", "D) <Option 4>".
        Please Please Remember to provide perfect and suitable option where the option should not repeated more then once for entire questions.
        <context>
        {{context}}
        <context>
        """

        fill_in_blank_prompt = f"""
        Generate {num_questions} fill-in-the-blank questions based on the following context.
        Each question should contain a blank ("___") and provide the correct answer for each question in the format "Answer: <correct_answer>" after the question.
        Remember to provide perfect and suitable answers where the answer should not repeated more then once.
        <context>
        {{context}}
        <context>
        """

        col1, col2 = st.columns([2, 1])
        with col1:
            st.write("Number of questions selected: ", num_questions)
        with col2:
            if st.button("Prepare Questions"):
                # Generate questions
                def generate_questions():
                    mcq_questions_formatted = []
                    fill_in_blank_questions_filtered = []
                    mcq_answers = []
                    fill_in_blank_answers = []

                    retriever = st.session_state.vectors.as_retriever()
                    retrieval_chain = create_retrieval_chain(retriever, document_chain)

                    for i, chunk in enumerate(st.session_state.documents):
                        context_text = chunk.page_content

                        # Generate multiple-choice questions
                        if len(mcq_questions_formatted) < num_questions:
                            mcq_response = retrieval_chain.invoke({"input": mcq_prompt.format(context=context_text)})
                            mcq_questions = mcq_response["answer"]

                            current_question = ""
                            options = []
                            mcq_answers = []
                            for line in mcq_questions.split("\n"):
                                if line.startswith("Q"):
                                    if current_question:
                                        random.shuffle(options)

                                        for j, option in enumerate(options):
                                            if option.endswith(mcq_answers[-1]):
                                                mcq_answers[-1] = chr(65 + j)  
                                                break
                                        mcq_questions_formatted.append(current_question.strip() + "\n" + "\n".join(options))

                                    current_question = f"**{line.strip()}**\n\n"
                                    options = []
                                elif line.startswith(("A)", "B)", "C)", "D)")):
                                    options.append(f"- {line.strip()}")
                                    if line.startswith("A)"):
                                        mcq_answers.append(line[3:].strip())

                            # Handle the last question
                            if current_question and len(mcq_questions_formatted) < num_questions:
                                if options:
                                    random.shuffle(options)

                                    for j, option in enumerate(options):
                                        if option.endswith(mcq_answers[-1]):
                                            mcq_answers[-1] = chr(65 + j)  
                                            break

                                    mcq_questions_formatted.append(current_question.strip() + "\n" + "\n".join(options))

                        # Generate fill-in-the-blank questions
                        if len(fill_in_blank_questions_filtered) < num_questions:
                            fill_in_blank_response = retrieval_chain.invoke({"input": fill_in_blank_prompt.format(context=context_text)})
                            fill_in_blank_questions = fill_in_blank_response["answer"].split("\n")

                            for line in fill_in_blank_questions:
                                if "Answer:" in line:
                                    question_part = line.split("Answer:")[0].strip()
                                    answer_part = line.split("Answer:")[1].strip()
                                    
                                    question_part = question_part.rstrip("(")  

                                    answer_part = answer_part.rstrip(").")  
                                    
                                    if "___" in question_part:
                                        fill_in_blank_questions_filtered.append(question_part)
                                        fill_in_blank_answers.append(answer_part)

                        # Stop if we have enough questions
                        if len(mcq_questions_formatted) >= num_questions and len(fill_in_blank_questions_filtered) >= num_questions:
                            break

                    while len(mcq_questions_formatted) < num_questions:
                        mcq_questions_formatted.append("MCQ Placeholder Question")
                        mcq_answers.append("Unknown")

                    while len(fill_in_blank_questions_filtered) < num_questions:
                        fill_in_blank_questions_filtered.append("Fill-in-the-Blank Placeholder Question")
                        fill_in_blank_answers.append("Unknown")

                    st.session_state.exam_questions["mcq"] = mcq_questions_formatted
                    st.session_state.exam_questions["fill_in_blank"] = fill_in_blank_questions_filtered
                    st.session_state.exam_questions["mcq_answers"] = mcq_answers
                    st.session_state.exam_questions["fill_in_blank_answers"] = fill_in_blank_answers

                generate_questions()
                st.session_state.questions_generated = num_questions

        # Display exam questions
        user_answers = {}

        if question_type == "All Questions":
            st.subheader("Multiple Choice Questions")
            for i, question in enumerate(st.session_state.exam_questions["mcq"], 1):
                if question.strip():  
                    st.markdown(question)
                    user_answers[f"mcq_{i}"] = st.selectbox(f" ", [f"Select the Option Q{i}", "A", "B", "C", "D"], key=f"mcq_{i}")

            st.subheader("Fill-in-the-Blank Questions")
            for i, question in enumerate(st.session_state.exam_questions["fill_in_blank"], 1):
                if question.strip() and "___" in question:  
                    st.markdown(question)
                    
                    user_answers[f"fib_{i}"] = st.text_input(f"Enter your answer here", key=f"fib_{i}")

        if question_type == "Multiple Choice Questions":
            st.subheader("Multiple Choice Questions")
            for i, question in enumerate(st.session_state.exam_questions["mcq"], 1):
                if question.strip():  
                    st.markdown(question)
                    user_answers[f"mcq_{i}"] = st.selectbox(f" ", [f"Select the Option Q{i}", "A", "B", "C", "D"], key=f"mcq_{i}")

        if question_type == "Fill-in-the-Blank Only":
            st.subheader("Fill-in-the-Blank Questions")
            for i, question in enumerate(st.session_state.exam_questions["fill_in_blank"], 1):
                if question.strip() and "___" in question:  
                    st.markdown(question)
                    
                    user_answers[f"fib_{i}"] = st.text_input(f"Enter your answer here", key=f"fib_{i}")

        # Submit button for answers
        if st.button("Submit Answers"):
            st.subheader("Results")

            max_questions = num_questions

            # Ensure similar handling for MCQs
            if question_type == "All Questions" or question_type == "Multiple Choice Questions":
                st.subheader("Multiple Choice Questions Results")
                for i, correct_answer in enumerate(st.session_state.exam_questions["mcq_answers"], 1):
                    if i > max_questions:
                        break  

                    user_answer = user_answers.get(f"mcq_{i}", "").strip()  
                    if user_answer == correct_answer:
                        st.markdown(f"**Q{i}:** Correct! ✅")
                    else:
                        st.markdown(f"**Q{i}:** Wrong ❌ (Correct answer: {correct_answer})")            
            
            # Check Fill-in-the-Blank answers
            if question_type == "All Questions" or question_type == "Fill-in-the-Blank Only":
                st.subheader("Fill-in-the-Blank Results")
                for i, correct_answer in enumerate(st.session_state.exam_questions["fill_in_blank_answers"], 1):
                    if i > max_questions:
                        break  

                    user_answer = user_answers.get(f"fib_{i}", "").strip()
                    if user_answer.lower() == correct_answer.lower():
                        st.markdown(f"**Q{i}:** Correct! ✅")
                    else:
                        st.markdown(f"**Q{i}:** Wrong ❌ (Correct answer: {correct_answer})")

# elif option == "Question Paper":
#     st.title("Question Paper Section")
#     st.write("Content for the Question Paper section goes here.")

