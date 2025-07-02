import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

# Function to get response from LLaMA 2 model
def getLLamaresponse(input_text, no_words, blog_style):
    llm = CTransformers(
        model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
        model_type='llama',
        config={
            'max_new_tokens': 256,
            'temperature': 0.01
        }
    )

    template = """
    You are a professional blog writer.

    Write a clear, engaging, and informative blog in around {no_words} words for a {blog_style} audience on the topic: "{input_text}".

    Avoid labeling sections like intro/body/conclusion. Make it a natural, flowing article with a smooth structure and human tone. Keep the language simple and avoid technical jargon unless necessary.
    """

    prompt = PromptTemplate(
        input_variables=["blog_style", "input_text", "no_words"],
        template=template
    )

    return llm.invoke(prompt.format(
        blog_style=blog_style,
        input_text=input_text,
        no_words=no_words
    ))



# Streamlit Page Configuration
st.set_page_config(
    page_title="Short Blog Generator",
    page_icon="✍️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Header Section
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>✍️ Short Blog Generator</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; font-size: 18px;'>Craft concise, engaging blogs in seconds using AI</p>",
    unsafe_allow_html=True
)
st.image("assets/header.png", use_container_width=True)
st.markdown("---")

# Input Form
with st.container():
    st.subheader("📄 Blog Input")
    input_text = st.text_input("📌 Enter the blog topic", placeholder="e.g., The Future of Electric Vehicles")

    col1, col2 = st.columns(2)
    with col1:
        no_words = st.number_input('🔢 Approximate word count', min_value=50, max_value=250, step=50, value=150)
    with col2:
        blog_style = st.selectbox(
            '🎯 Target Audience',
            ['Researchers', 'Data Scientist', 'Common People']
        )

    st.markdown("")
    submit = st.button("🚀 Generate Blog")

# Output Section
if submit:
    if input_text.strip() == "":
        st.warning("⚠️ Please enter a topic to generate your blog.")
    else:
        with st.spinner("🧠 Generating a short, impactful blog for you..."):
            response = getLLamaresponse(input_text, no_words, blog_style)
        st.markdown("---")
        st.subheader("📝 Your AI-Generated Blog")
        st.write(response.strip())
        st.balloons()
        st.toast("✅ Blog created successfully!", icon="✍️")
