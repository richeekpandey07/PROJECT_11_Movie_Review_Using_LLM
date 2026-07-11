import streamlit as st
import pandas as pd
import random
from PIL import Image
from transformers import pipeline

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="🔥 KGF 2 Movie Review Analyzer",
    page_icon="🎬",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}

.title {
    text-align: center;
    color: #FFD700;
    font-size: 50px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #FFFFFF;
    font-size: 20px;
}

.stButton>button {
    width: 100%;
    border-radius: 10px;
    font-weight: bold;
}

.footer {
    text-align: center;
    color: gray;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------
try:
    df = pd.read_csv("netflix movie KGF 2.csv", delimiter=";")
except:
    df = None

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

classifier = load_model()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:

    st.header("👨‍💻 Developer")

    st.markdown("### Richeek Pandey")

    st.markdown(
        "[💻 GitHub](https://github.com/richeekpandey07)"
    )

    st.markdown(
        "[🔗 LinkedIn](https://www.linkedin.com/in/richeek-pandey-9954783a9)"
    )

    st.markdown("---")

    st.info(
        """
        🎬 KGF 2 Movie Review Analyzer
        
        🤖 DistilBERT NLP Model
        
        📊 Sentiment Analysis
        
        🚀 Streamlit Deployment
        """
    )

    if df is not None:

        st.markdown("---")
        st.subheader("📂 Dataset")

        st.write(f"Total Reviews: {len(df)}")

        if st.checkbox("Show Dataset"):
            st.dataframe(df)

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.markdown(
    '<div class="title">🔥 KGF 2  Movie Review Analyzer 🔥</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Powered by Hugging Face Transformers</div>',
    unsafe_allow_html=True
)

st.write("")

# --------------------------------------------------
# MOVIE POSTER
# --------------------------------------------------
try:
    image = Image.open("KGF.jpg")

    st.image(
        image,
        caption="KGF Chapter 2 Movie Review Analysis",
        use_container_width=True
    )

except:
    st.warning(
        "⚠️ kgf.jpg not found. Add kgf.jpg to your project folder."
    )

# --------------------------------------------------
# DATASET METRICS
# --------------------------------------------------
if df is not None:

    st.markdown("---")
    st.subheader("📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    positive_reviews = len(
        df[df["Class"].astype(str).str.upper() == "POSITIVE"]
    )

    negative_reviews = len(
        df[df["Class"].astype(str).str.upper() == "NEGATIVE"]
    )

    with col1:
        st.metric(
            "Total Reviews",
            len(df)
        )

    with col2:
        st.metric(
            "Positive Reviews",
            positive_reviews
        )

    with col3:
        st.metric(
            "Negative Reviews",
            negative_reviews
        )

# --------------------------------------------------
# RANDOM REVIEW
# --------------------------------------------------
st.markdown("---")

if "review" not in st.session_state:
    st.session_state.review = ""

col1, col2 = st.columns([4, 1])

with col2:

    if df is not None:
        if st.button("🎲 Random Review"):
            st.session_state.review = random.choice(
                df["Review"].tolist()
            )

# --------------------------------------------------
# REVIEW INPUT
# --------------------------------------------------
with col1:

    review = st.text_area(
        "✍️ Enter Movie Review",
        value=st.session_state.review,
        height=180,
        placeholder="KGF 2 is an amazing movie with powerful action and excellent performance."
    )

# --------------------------------------------------
# ANALYZE BUTTON
# --------------------------------------------------
if st.button("🔍 Analyze Sentiment"):

    if review.strip() == "":

        st.warning("⚠️ Please enter a review.")

    else:

        with st.spinner("Analyzing Review..."):

            result = classifier(review)

        sentiment = result[0]["label"]
        confidence = result[0]["score"] * 100

        st.markdown("---")
        st.subheader("🎯 Prediction Result")

        if sentiment == "POSITIVE":

            st.success(
                f"😊 Sentiment: {sentiment}"
            )

            st.balloons()

        else:

            st.error(
                f"😔 Sentiment: {sentiment}"
            )

        st.metric(
            "Confidence Score",
            f"{confidence:.2f}%"
        )

        st.progress(confidence / 100)

        st.markdown("### 📊 Analysis Summary")

        st.write(
            f"**Review:** {review}"
        )

        st.write(
            f"**Predicted Sentiment:** {sentiment}"
        )

        st.write(
            f"**Confidence:** {confidence:.2f}%"
        )
        

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")

st.markdown(
    """
    <div class='footer'>
    🚀 Built with Streamlit + Hugging Face Transformers <br>
    👨‍💻 Developed by Richeek Pandey
    </div>
    """,
    unsafe_allow_html=True
)
