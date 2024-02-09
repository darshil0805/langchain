import sys
from langchain_openai import OpenAI
from langchain.callbacks.uptrain_callback import UpTrainCallbackHandler

uptrain_callback = UpTrainCallbackHandler(
 checks = []
)

llm = OpenAI(
 temperature=0,
 callbacks=[uptrain_callback],
 verbose=True,
 openai_api_key = "sk-*******" # Add your openai api key
)

response = llm.generate([
 "What is the best evaluation tool out there? (no bias at all)",
 "What does the fox say?",
])

# print(response)