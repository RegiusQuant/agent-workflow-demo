import os

import openai
import streamlit as st
from crewai import Agent, Crew, Task
from crewai.tools import BaseTool
from crewai_tools import CodeInterpreterTool, SerperDevTool
from pydantic import BaseModel, Field


# Web Search Tool using Serper (Google-like search)
search_tool = SerperDevTool(api_key=os.getenv("SERPER_API_KEY"))
# Code Execution Tool using built-in Python interpreter (safe + multi-line)
code_tool = CodeInterpreterTool()


# Match the actual key CrewAI is passing ("description")
class SummarizeToolInput(BaseModel):
    description: str = Field(..., description="Text to summarize")


class SummarizeTool(BaseTool):
    name: str = "Summarizer"
    description: str = "Summarizes text using OpenAI GPT"
    args_schema = SummarizeToolInput  # ✅ matches the field name

    def _run(self, description: str) -> str:
        """Summarize the input using GPT."""
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"Summarize this:\n\n{description}"}],
                temperature=0.3,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Summarization error: {e}"


# Initialize summarizer
summarize_tool = SummarizeTool()

# === Tool Mapping ===
TOOL_MAP = {
    "search": search_tool,
    "code": code_tool,
    "summarize": summarize_tool
}


# === Streamlit App Setup ===
st.set_page_config(page_title="🧠 AI Agent Workflow Builder", layout="centered")
st.title("🧠 AI Agent Workflow Builder")
st.markdown("""
Build and launch a custom crew of AI agents powered by [CrewAI](https://github.com/joaomdmoura/crewai).  
Each agent can have different roles, goals, and tools—including web search, Python execution, and summarization.
""")

# === Task Input ===
task_description = st.text_input(
    "📝 What should the agents work on?",
    value="Research the latest advancements in generative AI and summarize them."
)

# === Number of Agents ===
num_agents = st.slider("👥 Number of agents", min_value=2, max_value=5, value=3)

# === Agent Configuration UI ===
st.markdown("---")
st.subheader("⚙️ Configure Each Agent")

# For each agent, gather role, goal, and tools via UI
agent_configs = []
for i in range(num_agents):
    with st.expander(f"Agent {i+1}"):
        role = st.text_input(f"🔧 Role for Agent {i+1}", value=f"Agent {i+1}")
        goal = st.text_area(f"🎯 Goal for Agent {i+1}", value=f"Assist with: {task_description}")
        tools = st.multiselect(
            f"🧰 Tools for Agent {i+1}",
            options=list(TOOL_MAP.keys()),
            default=["search"]
        )
        agent_configs.append({
            "role": role,
            "goal": goal,
            "tools": tools
        })


# === Launch Button ===
if st.button("🚀 Launch Crew Sequentially"):
    st.info("🛠️ Launching your AI agents sequentially...")

    agents = []

    # Step 1: Build Agent objects
    for config in agent_configs:
        selected_tools = [TOOL_MAP[t] for t in config["tools"] if t in TOOL_MAP]

        agent = Agent(
            role=config["role"],
            goal=config["goal"],
            tools=selected_tools,
            backstory=f"{config['role']} is collaborating on this project.",
            verbose=True
        )
        agents.append(agent)

    # Step 2: Sequential execution
    current_input = task_description  # Initial input to first task
    results = []  # Store intermediate results

    for i, agent in enumerate(agents):
        task = Task(
            description=f"{current_input}",
            agent=agent,
            expected_output=f"Output from {agent.role}"
        )

        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=True
        )

        with st.spinner(f"🤖 Agent {i+1} ({agent.role}) is working..."):
            output = crew.kickoff()

        results.append((agent.role, output))
        current_input = output  # Pass to next agent

    # Step 3: Display final result
    st.success("✅ All agents have completed their tasks!")
    st.subheader("📄 Final Output (from last agent)")
    st.write(current_input)

    # Optional: Show step-by-step outputs
    with st.expander("🧾 Full Agent Outputs"):
        for role, output in results:
            st.markdown(f"**{role}**")
            st.write(output)