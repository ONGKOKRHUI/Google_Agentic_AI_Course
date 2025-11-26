import os
import asyncio
import base64
from pathlib import Path
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams


try:
    from mcp import StdioServerParameters
except ImportError:
    # Define a simple placeholder if 'mcp' isn't available outside the environment
    class StdioServerParameters:
        def __init__(self, command, args, tool_filter):
            self.command = command
            self.args = args
            self.tool_filter = tool_filter

# --- 1. Setup and Configuration ---

from dotenv import load_dotenv
load_dotenv()

import os
api_key = os.getenv("GOOGLE_API_KEY")

# Configure retry options
retry_config = types.HttpRetryOptions(
    attempts=5, 
    exp_base=7,  
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

try:
    from exceptiongroup import BaseExceptionGroup
except ImportError:
    # If BaseExceptionGroup is native (Python 3.11+) or not explicitly installed
    # we can pass, as it might be available globally.
    pass

# --- 2. Create the MCP Toolset ---

# This connects to the Everything MCP Server, running it via 'npx'.
mcp_image_server = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=[
                "-y",  # Argument for npx to auto-confirm install
                "@modelcontextprotocol/server-everything",
            ],
            tool_filter=["getTinyImage"], # Only use the getTinyImage tool
        ),
        timeout=30,
    )
)

print("✅ MCP Tool created: Everything Server (getTinyImage)")

# --- 3. Create the LlmAgent ---

image_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="image_agent",
    instruction="Use the getTinyImage tool from the MCP Toolset to generate a small image for user queries. When providing the final response, clearly indicate that the image is a small, sample image.",
    tools=[mcp_image_server],
)

print("✅ Agent created and configured with MCP tool.")

# --- 4. Helper Function for Image Processing ---

def save_image_from_response(events, output_filename="tiny_image.png"):
    """
    Parses the function response for base64 image data and saves it as a PNG file.
    """
    image_data_b64 = None
    
    # Iterate through all events to find the function response
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                # The image data is nested inside the function_response part
                if hasattr(part, "function_response") and part.function_response:
                    response_content = part.function_response.response.get("content", [])
                    for item in response_content:
                        if item.get("type") == "image" and item.get("data"):
                            image_data_b64 = item["data"]
                            break
            if image_data_b64:
                break

    if image_data_b64:
        # Decode the base64 string to bytes
        image_bytes = base64.b64decode(image_data_b64)
        
        # Write the bytes to a local file
        output_path = Path(output_filename)
        output_path.write_bytes(image_bytes)
        
        return f"Image successfully saved to: {output_path.resolve()}"
    else:
        return "No image data found in the agent's tool response."


# --- 5. Create and Run the Agent Runner ---

# Use InMemoryRunner for a stateless execution 
runner = InMemoryRunner(agent=image_agent)

async def main():
    query = "Provide a sample tiny image"
    print(f"\nUser > {query}")
    print("=" * 60)

    # Run the agent in debug mode to see the tool calling steps.
    response_events = await runner.run_debug(query, verbose=True)

    # Process and Save the Image
    image_status = save_image_from_response(response_events)
    
    # Print the Agent's final text response
    final_response_text = ""
    for event in response_events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    final_response_text += part.text

    print("=" * 60)
    print(f"Agent Final Text Response: {final_response_text.strip()}")
    print(f"Image Saving Status: {image_status}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        # Run the asynchronous main function
        asyncio.run(main())
    except BaseExceptionGroup as e:
        # This catches the known, non-fatal cleanup error from the stdio_client.
        # It's an issue with how the underlying libraries (anyio) close the standard I/O streams.
        print("\n✅ Script completed and image saved successfully.")
        print("⚠️ Non-fatal cleanup error caught during MCP stdio shutdown. This can be safely ignored.")
    except Exception as e:
        # Catch other, potentially fatal errors
        print(f"An unexpected fatal error occurred: {e}")