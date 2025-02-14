import os
import json
from dotenv import load_dotenv
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain import LLMChain, PromptTemplate

# Load the environment variables
load_dotenv()
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")

model_name = "meta-llama/Llama-3-1B-Instruct"
pipe = pipeline("text-generation", model=model_name, max_new_tokens=512, use_auth_token=huggingface_api_key)
llm = HuggingFacePipeline(pipeline=pipe)

prompt_template = """
You are an AI that extracts structured event data from raw conversation text.
Below are the event types and their required JSON schema definitions:

DetectionEvent:
{
    "eventType": "detection",
    "timestamp": "ISO8601 string (e.g., 2024-03-20T14:30:00.000Z)",
    "objectId": "string",
    "confidence": float,
    "position": [float, float, float],
    "classification": "string",
    "boundingBox": {
      "x": int,
      "y": int,
      "width": int,
      "height": int
    }
}

LocationChangedEvent:
{
    "eventType": "locationChanged",
    "timestamp": "ISO8601 string",
    "objectId": "string",
    "previousPosition": [float, float, float],
    "newPosition": [float, float, float],
    "velocity": [float, float, float]
}

SpeechEvent:
{
    "eventType": "speech",
    "timestamp": "ISO8601 string",
    "speakerId": "string",
    "content": "string",
    "confidence": float,
    "language": "string",
    "duration": float
}

SupportNeededEvent:
{
    "eventType": "supportNeeded",
    "timestamp": "ISO8601 string",
    "requestId": "string",
    "priority": "string",
    "location": [float, float, float],
    "description": "string",
    "requesterType": "string",
    "status": "string"
}

Given the conversation text below, extract all events that occur and output a JSON array (list) where each element is one event object exactly matching one of the above schemas. Do not output any additional text outside of the JSON.

Conversation Text:
{conversation_text}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["conversation_text"]
)

chain = LLMChain(llm=llm, prompt=prompt)

conversation_text = """
At 14:30:00, a detection event occurred for person_123 with a confidence of 0.95. The position was [1.5, 2.0, -0.5] and the bounding box was x:100, y:150, width:50, height:80.
Shortly after, at 14:30:05, robot_A1 changed location from [0.0, 0.0, 0.0] to [1.2, 0.0, 3.4] with a velocity of [0.5, 0.0, 0.8].
At 14:31:00, user_456 said, "Hello, can you help me find the exit?" with a confidence of 0.88, speaking in en-US and lasting 2.5 seconds.
Finally, at 14:32:00, support request support_789 was raised with high priority at location [12.5, 8.0, -1.5] for an elderly person needing assistance. The requester was a visitor and the status is pending.
"""

response = chain.run(conversation_text=conversation_text)

print("Raw LLM Response:")
print(response)

try:
    events = json.loads(response)
    print("\nParsed JSON Events:")
    for event in events:
        print(event)
except json.JSONDecodeError as e:
    print("Failed to parse JSON:", e)
