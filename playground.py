import openai
from phi.agent import Agent
import phi.api
from phi.model.openai import OpenAIChat
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import os
import phi
from phi.playground import Playground, serve_playground_app

load_dotenv()

phi.api=os.getenv("PHI_API_KEY")


web_search_agent = Agent(
    name='web_search_agent',
    role='search the web for the information',
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Always include the source of the information in the response.", "tokens per minute (TPM): Limit 6000"],
    show_tool_calls=True,
    markdown=True,
)


## Funancial Agent

financial_agent = Agent(
    name = "Finance AI Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,
                      company_news=True, company_info=True, technical_indicators=True, historical_prices=True,),
    ],
    instructions=["USe tables to display the data.",  "tokens per minute (TPM): Limit 6000"],
    show_tool_calls=True,
    markdown=True,
    
)

app = Playground(agents=[web_search_agent, financial_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app",reload=True) 