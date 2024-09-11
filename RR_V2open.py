import numpy as np
import streamlit as st
import plotly.graph_objs as go

# 1. Enhanced Multi-Agent Perceptions Visualization with Plotly
def visualize_multi_agent_perceptions():
    num_agents = 5
    num_points = 50
    colors = ['red', 'green', 'blue', 'orange', 'purple']

    fig = go.Figure()

    for i in range(num_agents):
        x = np.random.rand(num_points)
        y = np.random.rand(num_points)
        z = np.random.rand(num_points)

        # Create scatter plots for each agent
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers+lines',
            marker=dict(size=5, color=colors[i], opacity=0.8),
            name=f'Agent {i+1}',
            line=dict(color=colors[i], width=2, dash='dash')
        ))

    fig.update_layout(
        title="Multi-Agent Perceptions in 3D Feature Space",
        scene=dict(
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",
            zaxis_title="Feature 3"
        ),
        legend=dict(x=0, y=1.0),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    st.plotly_chart(fig)

# 2. Enhanced Emotion-based Reinforcement Learning Visualization with Plotly
def visualize_emotion_reinforcement_learning():
    states = ['S1', 'S2', 'S3', 'S4']
    actions = ['A1', 'A2']
    emotions = ['Happy', 'Neutral', 'Sad']

    fig = go.Figure()

    for i, emotion in enumerate(emotions):
        Q = np.random.rand(len(states), len(actions))
        heatmap = go.Heatmap(
            z=Q,
            x=actions,
            y=states,
            colorscale='Viridis',
            name=f'Q-values for {emotion}'
        )
        fig.add_trace(heatmap)

    fig.update_layout(
        title="Q-Values for Emotion-based Reinforcement Learning",
        xaxis_title="Actions",
        yaxis_title="States",
        margin=dict(l=0, r=0, b=0, t=40)
    )
    st.plotly_chart(fig)

# 3. Enhanced Quantum Decision Making Visualization with Plotly
def visualize_quantum_decision_making():
    def quantum_amplitude(x, y, t):
        return np.sin(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * t)**2
    
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)

    fig = go.Figure()

    for t in [0, 0.25, 0.5]:
        Z = quantum_amplitude(X, Y, t)
        surface = go.Surface(z=Z, x=X, y=Y, colorscale='Plasma')
        fig.add_trace(surface)

    fig.update_layout(
        title="Quantum Decision Making Over Time",
        scene=dict(
            xaxis_title="Decision Axis 1",
            yaxis_title="Decision Axis 2",
            zaxis_title="Quantum Amplitude"
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    st.plotly_chart(fig)

# Streamlit App
def main():
    st.title("3D Visualizations of Perceptions, Reinforcement Learning, and Quantum Decisions")

    st.subheader("Multi-Agent Perceptions")
    st.write("""
        This visualization represents how multiple agents move through a 3D feature space.
        Each agent's movement is represented by points and dashed lines showing their trajectory.
    """)
    visualize_multi_agent_perceptions()

    st.subheader("Emotion-based Reinforcement Learning")
    st.write("""
        This heatmap shows Q-values associated with different emotional states (Happy, Neutral, Sad).
        The Q-values represent the expected rewards for taking actions in specific states.
    """)
    visualize_emotion_reinforcement_learning()

    st.subheader("Quantum Decision Making")
    st.write("""
        This 3D surface plot shows quantum decision amplitudes over time. Each surface represents the quantum state
        at different time steps, illustrating how decisions evolve in a quantum system.
    """)
    visualize_quantum_decision_making()

if __name__ == "__main__":
    main()


