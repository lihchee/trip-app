import streamlit as st
from google import genai
from google.genai import types
import polars as pl
import tempfile
import os
import json
from pydantic import BaseModel, Field
from typing import List, Optional

# 1. Define what the output structured JSON should look like
class TravelEvent(BaseModel):
    event_type: str = Field(description="Type of event: 'Flight', 'Hotel', 'Train', or 'Activity'")
    start_datetime: Optional[str] = Field(description="Start time in ISO 8601 format (e.g., 2026-12-28T01:10:00). Leave null if unknown.")
    end_datetime: Optional[str] = Field(description="End time in ISO 8601 format (e.g., 2026-12-28T09:00:00). Leave null if unknown.")
    start_location: Optional[str] = Field(description="Departure airport code, train station, or hotel name/address.")
    end_location: Optional[str] = Field(description="Arrival airport or station. Null for hotels/activities.")
    booking_reference: Optional[str] = Field(description="PNR, booking ID, or confirmation code.")
    raw_details: Optional[str] = Field(description="Key details: e.g., 'Seats 27A-D, 30kg baggage, Gyudon meals'.")

class TripItinerary(BaseModel):
    events: List[TravelEvent]

# 2. Build the Webpage UI
st.set_page_config(page_title="TripTry", page_icon="✈️", layout="wide")
st.title("✈️ TripTry: Personal Itinerary Extractor")
st.markdown("Upload a PDF booking confirmation to generate a structured timeline.")

# 2.5 Get the API Key securely (Checks Streamlit Secrets or sidebar fallback)
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("Google AI Studio API Key", type="password")
        st.markdown("[Get a free API key here](https://aistudio.google.com/)")

# File Uploader
uploaded_file = st.file_uploader("Drop your PDF booking receipt here", type=["pdf", "txt"])

# 3. Process the File
if uploaded_file and api_key:
    if st.button("Extract Itinerary"):
        client = genai.Client(api_key=api_key)
        
        with st.spinner("Parsing document with Gemini 3.8 Flash..."):
            try:
                # Save the uploaded file temporarily so Gemini can read it
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_path = temp_file.name

                # Upload to Google's temporary File API
                gemini_file = client.files.upload(file=temp_path)
                
                prompt = """
                You are an expert travel assistant. Read this booking confirmation and extract 
                every travel event (flights, check-ins, check-outs) into the required JSON schema.
                Ensure all dates are converted to standard ISO 8601 format.
                """
                
                # Generate content using the new SDK and current Gemini 3.8 Flash model
                response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=[gemini_file, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=TripItinerary,
                        temperature=0.1 
                    )
                )

                # Clean up temporary files
                client.files.delete(name=gemini_file.name)
                os.remove(temp_path)

                # 4. Display the Results
                extracted_data = json.loads(response.text)
                
                if extracted_data.get("events"):
                    df = pl.DataFrame(extracted_data["events"])
                    df = df.sort("start_datetime")
                    
                    st.success("Extraction Complete!")
                    st.subheader("Your Master Timeline")
                    st.dataframe(
                        df.to_pandas(), 
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    with st.expander("View Raw JSON Output"):
                        st.json(extracted_data)
                else:
                    st.warning("No travel events found in this document.")

            except Exception as e:
                st.error(f"An error occurred: {e}")

elif uploaded_file and not api_key:
    st.warning("Please enter your Gemini API key in the sidebar or save it in Streamlit Secrets to continue.")
