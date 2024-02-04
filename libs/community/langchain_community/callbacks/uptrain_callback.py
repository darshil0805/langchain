"""UpTrain's Callback Handler."""
#Langchain Community
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union


if TYPE_CHECKING:
    import uptrain

def import_uptrain():
    try:
        import uptrain
    except ImportError as e:
        raise ImportError(
            "To use the UpTrainCallbackHandler, you need the"
            "`uptrain` package. Please install it with"
            "`pip install uptrain`.",
            e
        )

    return uptrain

class UpTrainCallbackHandler(BaseCallbackHandler):
    """Callback Handler that logs to uptrain.

    Args:
        BaseCallbackHandler (_type_): _description_
    """

    REPO_URL: str = "https://github.com/uptrain-ai/uptrain"
    ISSUES_URL: str = f"{REPO_URL}/issues"
    DOCS_URL: str = "https://docs.uptrain.ai"

    def __init__(
            self,
            checks,
            project_name: Optional[str] = None,
            retrieval = False
            ) -> None:
        """Initializes the `UpTrainCallbackHandler`."""
        super().__init__()

        uptrain = import_uptrain()

        #from uptrain.framework import Settings
        import os

        # Set uptrain variables
        self.project_name = project_name
        self.checks = checks
        self.retrieval_flag = retrieval
        # Set up uptrain settings
        # self.uptrain_settings = Settings(
        #     uptrain_access_token=os.environ["UPTRAIN_API_KEY"],
        #     openai_api_key=os.environ["OPENAI_API_KEY"],
        # )
        

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Store the prompts"""
        if(self.retrieval_flag == False):
            self.prompts = prompts
        else:
            self.context = prompts

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Do nothing when a new token is generated."""
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Log records to uptrain when an LLM ends."""
        
        if(self.retrieval_flag == False):
            from uptrain import APIClient, Evals
            
            data = [] 
            len_queries = len(self.prompts)
            for i in range(len_queries):
                data.append({'question':self.prompts[i],'response':response.generations[i].text})

            UPTRAIN_API_KEY = 'add uptrain api key'
            client = APIClient(uptrain_api_key=UPTRAIN_API_KEY)
            results = client.log_and_evaluate(
                project_name= self.project_name,
                data=data,
                checks=[Evals.RESPONSE_RELEVANCE]
            )
        
    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing when LLM outputs an error."""
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        self.prompts = inputs
        """Do nothing when chain starts"""
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Do nothing when chain ends."""
        from uptrain import APIClient, Evals
        data = [] 
        len_queries = len(self.prompts)
        for i in range(len_queries):
            data.append({'question':self.prompts[i],'context':self.context[i],'response':outputs.generations[i].text})

        UPTRAIN_API_KEY = 'add uptrain api key'
        client = APIClient(uptrain_api_key=UPTRAIN_API_KEY)
        results = client.log_and_evaluate(
            project_name= self.project_name,
            data=data,
            checks=[Evals.RESPONSE_RELEVANCE]
        )
    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing when LLM chain outputs an error."""
        pass

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Do nothing when tool starts."""
        pass

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Do nothing when agent takes a specific action."""
        pass

    def on_tool_end(
        self,
        output: str,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Do nothing when tool ends."""
        pass

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing when tool outputs an error."""
        pass

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Do nothing"""
        pass

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Do nothing"""
        pass

