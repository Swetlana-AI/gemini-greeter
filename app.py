import os
import time
from flask import Flask, render_template_string
from google import genai
from google.genai import types

# --- Configuration ---
# The API key is expected to be provided via an environment variable (GEMINI_API_KEY)
API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"
app = Flask(__name__)

# Basic Exponential Backoff for API calls
def call_gemini_with_backoff(client, contents, system_instruction, tools, max_retries=5):
    """Handles Gemini API call with exponential backoff."""
    for attempt in range(max_retries):
        try:
            print(f"Attempting API call (Attempt {attempt + 1})...")
            # Use Google Search as a tool for grounding
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=tools
            )

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=config,
            )
            print("API call successful.")
            return response
        except Exception as e:
            print(f"API call failed on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                sleep_time = 2 ** attempt
                print(f"Waiting {sleep_time} seconds before retrying...")
                time.sleep(sleep_time)
            else:
                return None # Return None after all retries fail
    return None

@app.route('/')
def home():
    if not API_KEY:
        # Instruction for the user if the key is missing (e.g., in local testing)
        return render_template_string("""
            <div style="padding: 40px; text-align: center; font-family: sans-serif;">
                <h1 style="color: #EF4444;">API Key Missing</h1>
                <p>The <code>GEMINI_API_KEY</code> environment variable is not set.</p>
                <p>Please ensure you set the environment variable or, if deploying, that the platform is injecting it.</p>
                <p style="color: #6B7280; font-size: 14px;">(The key is left blank in this file to be securely injected by the environment.)</p>
            </div>
        """)

    try:
        # Initialize the client with the environment variable key
        client = genai.Client(api_key=API_KEY)
    except Exception as e:
        return render_template_string(f"<h1 style='color: #EF4444;'>Client Initialization Error</h1><p>{e}</p>")

    # --- LLM Query Setup ---
    system_prompt = "You are a friendly, witty, and highly helpful assistant running on a new web service. Your task is to greet the user and provide a single, fascinating, and up-to-date piece of information about the world of technology or science, grounded by Google Search."
    user_query = "Give me one amazing fact about the current state of AI or space exploration."

    # Tools configuration (enables Google Search grounding)
    tools = [{"google_search": {}}]

    # Contents (the user's prompt)
    contents = [types.Content(parts=[types.Part.from_text(user_query)])]

    # Make the API call
    response = call_gemini_with_backoff(
        client,
        contents,
        system_prompt,
        tools
    )

    if response is None:
        return render_template_string(
            """
            <div style="padding: 40px; text-align: center; font-family: sans-serif;">
                <h1 style="color: #F59E0B;">API Call Failed</h1>
                <p>Could not get a response from the Gemini API after multiple retries.</p>
                <p>Please check your network connection and API key validity.</p>
            </div>
            """
        )

    # --- Process Response ---
    generated_text = response.text
    sources = []
    
    # Check for grounding metadata
    if response.candidates and response.candidates[0].grounding_metadata:
        grounding_metadata = response.candidates[0].grounding_metadata
        if grounding_metadata.grounding_attributions:
            sources = [
                {'uri': attr.web.uri, 'title': attr.web.title}
                for attr in grounding_metadata.grounding_attributions
                if attr.web and attr.web.uri and attr.web.title
            ]

    # --- HTML Rendering ---
    html_sources = ""
    if sources:
        list_items = "".join([
            f"<li class='text-sm text-gray-500'><a href='{s['uri']}' target='_blank' class='hover:underline text-blue-500'>{s['title']}</a></li>"
            for s in sources
        ])
        html_sources = f"""
            <div class="mt-8 pt-4 border-t border-gray-200">
                <p class="text-xs font-semibold uppercase text-gray-400 mb-2">Sources (Grounded by Google Search):</p>
                <ul class="list-disc list-inside space-y-1">{list_items}</ul>
            </div>
        """

    # Use Tailwind-like classes for a nice appearance
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Gemini Hello World App</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
        <style>
            body {{ font-family: 'Inter', sans-serif; background-color: #f7fafc; }}
        </style>
    </head>
    <body class="p-8">
        <div class="max-w-3xl mx-auto bg-white shadow-xl rounded-xl p-8 md:p-12 mt-10">
            <h1 class="text-3xl font-bold text-gray-800 mb-6 flex items-center">
                🤖 Gemini App Running on Doprax
            </h1>
            <div class="bg-blue-50 border-l-4 border-blue-400 text-blue-800 p-4 rounded-lg">
                <p class="font-medium text-lg">Response from Gemini:</p>
                <p class="mt-2 whitespace-pre-wrap">{generated_text}</p>
            </div>
            {html_sources}
            <div class="mt-10 text-center text-gray-400 text-sm">
                <p>This application was deployed using Python 3.12 and Flask.</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html_template

if __name__ == '__main__':
    # When running locally, Flask defaults to 127.0.0.1:5000
    # In a container environment, it must listen on 0.0.0.0 and use a standard port (like 8080)
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=True)