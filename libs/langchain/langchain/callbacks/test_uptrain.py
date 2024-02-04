import sys
#sys.path.append('/Users/darshiljariwala/Desktop/langchain/libs/langchain/langchain/callbacks')
print(sys.path)
#from langchain.callbacks import TrubricsCallbackHandler
from langchain.llms import OpenAI
from langchain.callbacks import UpTrainCallbackHandler

#from uptrain.operators import LanguageCritique, ResponseCompleteness, ResponseRelevance

# language_critique = LanguageCritique()
# response_completeness = ResponseCompleteness()

uptrain_callback = UpTrainCallbackHandler(
 checks = []
)

llm = OpenAI(
 temperature=0,
 callbacks=[uptrain_callback],
 verbose=True,
 openai_api_key = "add openai api key"
)

response = llm.generate([
 "What is the best evaluation tool out there? (no bias at all)",
 "What does the fox say?",
])

# print(response)