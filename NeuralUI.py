import streamlit as st
from Neural import get_references, get_answer
import base64

st.set_page_config(page_title="QA System", page_icon="🤖")

st.header("Interactive QA System")
user_query = st.text_area("Enter your question:")

if st.button("Get Answer"):
    references = get_references(user_query, radius=1)

    answer = get_answer(user_query, references)

    st.subheader("GenAI Curated Response:")
    st.markdown(f'<div style="border: 1px solid #ccc; padding: 10px; border-radius: 10px; background-color: #B3F2FA">{answer}</div>', unsafe_allow_html=True)

    st.subheader("References:")
    for ref in references:
        combined_text = f"{ref['text']}<br><br>Source: {ref['source']}"
        
        # Display the image in the first half and the combined text in the second half
        image_file = open("Thumsup.png", "rb")
        image_data = image_file.read()
        image_base64 = base64.b64encode(image_data).decode("utf-8")
        first_half = f'<div style="text-align: center;"><img id="clickable-image" src="data:image/png;base64,{image_base64}" alt="Image" style="width: 50%; height: 50%; cursor: pointer;"></div>'

        second_half = combined_text

        st.markdown(f"""
            <script>
                document.getElementById("clickable-image").addEventListener("click", function() {{
                    // Perform the desired functionality when the image is clicked
                    alert("Image clicked!");
                }});
            </script>
        """, unsafe_allow_html=True)

        st.markdown(f'<div style="border: 1px solid #ddd; margin-bottom: 15px; padding: 10px; border-radius: 10px; background-color: #DCE8EA;">'
                    f'<div style="float: left; width: 20%;">{first_half}</div>'
                    f'<div style="float: right; width: 80%;">{second_half}</div>'
                    f'<div style="clear: both;"></div>'
                    f'</div>', unsafe_allow_html=True)

