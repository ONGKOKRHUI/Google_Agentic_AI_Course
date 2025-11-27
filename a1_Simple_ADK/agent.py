from google.adk.agents import Agent
#from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
#from google.genai import types

print("✅ ADK components imported successfully.")

root_agent = Agent(
    name="helpful_assistant",
    model="gemini-2.5-flash-lite",
    description="A simple agent that can answer general questions.",
    instruction="You are a helpful assistant. Use Google Search for current info or if unsure.",
    tools=[google_search],
)



# print("✅ Root Agent defined.")

# runner = InMemoryRunner(agent=root_agent)

# print("✅ Runner created.")

# response = await runner.run_debug("What is Agent Development Kit from Google? What languages is the SDK available in?")