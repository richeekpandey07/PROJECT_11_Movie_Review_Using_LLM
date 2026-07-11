import streamlit as st
from transformers import pipeline

# Page Configuration
st.set_page_config(
    page_title="🎬 KGF 2 Movie Review Analyzer",
    page_icon="🎥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}

.title {
    text-align: center;
    color: #FFD700;
    font-size: 45px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #FFFFFF;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="title">🎬 KGF 2 Movie Review Analyzer</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">AI-Powered Sentiment Analysis using Hugging Face Transformers</p>',
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.header("👨‍💻 Developer")

    st.write("### Richeek Pandey")

    st.markdown(
        "[🔗 LinkedIn](https://www.linkedin.com/in/YOUR-LINKEDIN-ID)"
    )

    st.markdown(
        "[💻 GitHub](https://github.com/richeekpandey07)"
    )

    st.markdown("---")

    st.info(
        """
        This app uses DistilBERT NLP Model
        to analyze movie reviews and predict
        whether the sentiment is Positive or Negative.
        """
    )

# Load Model
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

classifier = load_model()

# Input Section
st.subheader("✍️ Enter Movie Review")

review = st.text_area(
    "Type your review here...",
    height=180,
    placeholder="KGF 2 is an amazing movie with powerful action and excellent performance."
)

# Predict Button
if st.button("🔍 Analyze Sentiment", use_container_width=True):

    if review.strip():

        with st.spinner("Analyzing Review..."):
            result = classifier(review)

        sentiment = result[0]["label"]
        confidence = result[0]["score"] * 100

        st.markdown("---")

        if sentiment == "POSITIVE":
            st.success(f"😊 Sentiment: {sentiment}")
            st.balloons()
        else:
            st.error(f"😔 Sentiment: {sentiment}")

        st.metric(
            label="Confidence Score",
            value=f"{confidence:.2f}%"
        )

        st.progress(float(confidence / 100))

        st.subheader("📊 Analysis Summary")

        st.write(f"**Review:** {review}")
        st.write(f"**Predicted Sentiment:** {sentiment}")
        st.write(f"**Confidence:** {confidence:.2f}%")

    else:
        st.warning("⚠️ Please enter a review first.")

st.markdown("---")
st.caption("🚀 Built with Streamlit + Hugging Face Transformers")
