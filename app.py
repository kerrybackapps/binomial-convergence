import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import plotly.graph_objects as go

# Configure page
st.set_page_config(page_title="Binomial Convergence", layout="wide")

# Enable iframe embedding
st.markdown("""
<script>
// Allow iframe embedding
if (window.location !== window.parent.location) {
    document.domain = document.domain;
}
// Remove X-Frame-Options restrictions
window.addEventListener('load', function() {
    if (window.parent !== window) {
        console.log('Running in iframe - embedding allowed');
    }
});
</script>
""", unsafe_allow_html=True)

# CSS for responsive iframe scaling
st.markdown("""
<style>
/* Make the entire app responsive */
.main .block-container {
    max-width: 100% !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}

/* Scale content to fit iframe */
.stApp {
    transform-origin: top left;
    width: 100%;
}

/* Responsive text scaling */
@media (max-width: 800px) {
    .main .block-container {
        font-size: 0.85rem;
    }
}
</style>
""", unsafe_allow_html=True)

def callBS(S, K, T, sigma, r, q):
    """Black-Scholes call option price"""
    def f(s):
        s = s if s != 0 else 1.0e-6
        d1 = (np.log(s / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return np.exp(-q * T) * s * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)

    if isinstance(S, list) or isinstance(S, np.ndarray):
        return np.array([f(s) for s in S])
    else:
        return f(S)


def putBS(S, K, T, sigma, r, q):
    """Black-Scholes put option price"""
    def f(s):
        s = s if s != 0 else 1.0e-6
        d1 = (np.log(s / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return np.exp(-r * T) * K * norm.cdf(-d2) - np.exp(-q * T) * s * norm.cdf(-d1)

    if isinstance(S, list) or isinstance(S, np.ndarray):
        return np.array([f(s) for s in S])
    else:
        return f(S)


def callEuropean(S, K, T, sigma, r, q, N=100):
    """European call option price using binomial model"""
    def f(s):
        s = s if s != 0 else 1.0e-6
        dt = T / N
        up = np.exp(sigma * np.sqrt(dt))
        down = 1 / up
        prob = (np.exp((r - q) * dt) - down) / (up - down)
        discount = np.exp(-r * dt)
        v = np.zeros(N + 1)
        x = s * up ** N
        v[0] = np.maximum(x - K, 0)

        for i in range(1, N + 1):
            x *= down * down
            v[i] = np.maximum(x - K, 0)

        for n in range(N - 1, -1, -1):
            x = s * up ** n
            v[0] = discount * (prob * v[0] + (1 - prob) * v[1])
            for i in range(1, n + 1):
                x *= down * down
                v[i] = discount * (prob * v[i] + (1 - prob) * v[i + 1])
        return v[0]

    if isinstance(S, list) or isinstance(S, np.ndarray):
        return np.array([f(s) for s in S])
    else:
        return f(S)


def putEuropean(S, K, T, sigma, r, q, N=100):
    """European put option price using binomial model"""
    def f(s):
        s = s if s != 0 else 1.0e-6
        dt = T / N
        up = np.exp(sigma * np.sqrt(dt))
        down = 1 / up
        prob = (np.exp((r - q) * dt) - down) / (up - down)
        discount = np.exp(-r * dt)
        v = np.zeros(N + 1)
        x = s * up ** N
        v[0] = np.maximum(K - x, 0)
        for i in range(1, N + 1):
            x *= down * down
            v[i] = np.maximum(K - x, 0)
        for n in range(N - 1, -1, -1):
            x = s * up ** n
            v[0] = discount * (prob * v[0] + (1 - prob) * v[1])
            for i in range(1, n + 1):
                x *= down * down
                v[i] = discount * (prob * v[i] + (1 - prob) * v[i + 1])
        return v[0]

    if isinstance(S, list) or isinstance(S, np.ndarray):
        return np.array([f(s) for s in S])
    else:
        return f(S)


def create_convergence_plots(K, sigma, r, q, T, S, maxN=50):
    """Create convergence plots for call and put options"""
    # Convert percentages to decimals
    sigma_decimal = sigma / 100
    r_decimal = r / 100
    q_decimal = q / 100

    # Create array of time steps (even numbers only)
    a = np.linspace(1, maxN, maxN) * 2

    # Calculate binomial call option values
    b_call = np.zeros(maxN)
    for i in range(maxN):
        b_call[i] = callEuropean(S, K, T, sigma_decimal, r_decimal, q_decimal, int(a[i]))

    # Calculate Black-Scholes call value
    bs_call = callBS(S, K, T, sigma_decimal, r_decimal, q_decimal)

    # Create call option plot
    fig_call = go.Figure()
    fig_call.add_trace(go.Scatter(
        x=a, y=b_call, 
        mode="markers",
        marker=dict(size=8, color='blue'),
        name="Binomial Value",
        hovertemplate="N = %{x}<br>Binomial value = $%{y:0.4f}<extra></extra>"
    ))
    fig_call.add_trace(go.Scatter(
        x=np.linspace(0, 2*maxN, 100),
        y=[bs_call] * 100,
        name="Black-Scholes",
        line=dict(color='red', dash='dot', width=2),
        hovertemplate="Black-Scholes value = $%{y:0.4f}<extra></extra>"
    ))

    # Calculate binomial put option values
    b_put = np.zeros(maxN)
    for i in range(maxN):
        b_put[i] = putEuropean(S, K, T, sigma_decimal, r_decimal, q_decimal, int(a[i]))

    # Calculate Black-Scholes put value
    bs_put = putBS(S, K, T, sigma_decimal, r_decimal, q_decimal)

    # Create put option plot
    fig_put = go.Figure()
    fig_put.add_trace(go.Scatter(
        x=a, y=b_put,
        mode="markers",
        marker=dict(size=8, color='blue'),
        name="Binomial Value",
        hovertemplate="N = %{x}<br>Binomial value = $%{y:0.4f}<extra></extra>"
    ))
    fig_put.add_trace(go.Scatter(
        x=np.linspace(0, 2*maxN, 100),
        y=[bs_put] * 100,
        name="Black-Scholes",
        line=dict(color='red', dash='dot', width=2),
        hovertemplate="Black-Scholes value = $%{y:0.4f}<extra></extra>"
    ))

    # Update layout for both plots
    for fig, title in [(fig_call, "European Call"), (fig_put, "European Put")]:
        fig.update_yaxes(title="Option Value")
        fig.update_xaxes(title="Number of Time Steps in Binomial Model")
        fig.update_layout(
            title={
                "text": title,
                "y": 0.95,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
                "font": {"size": 16}
            },
            yaxis_tickprefix="$", 
            yaxis_tickformat=",.4f",
            legend=dict(x=0.99, xanchor="right", y=0.99, yanchor="top"),
            height=400,
            showlegend=True
        )

    return fig_call, fig_put


# Streamlit UI


# Create input controls
col1, col2, col3 = st.columns(3)

with col1:
    K = st.slider("Strike Price ($)", min_value=0, max_value=100, value=50, step=1)
    T = st.slider("Years to Maturity", min_value=0.0, max_value=5.0, value=1.0, step=0.25)
    S = st.slider("Underlying Price ($)", min_value=0, max_value=100, value=60, step=1)

with col2:
    sigma = st.slider("Volatility (%)", min_value=0, max_value=80, value=40, step=1)
    q = st.slider("Dividend Yield (%)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
    maxN = st.slider("Maximum Time Steps", min_value=10, max_value=100, value=50, step=10)

with col3:
    r = st.slider("Risk-free Rate (%)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

st.markdown("---")

# Create and display plots
if st.button("Generate Plots") or True:  # Auto-generate on parameter change
    try:
        fig_call, fig_put = create_convergence_plots(K, sigma, r, q, T, S, maxN)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(fig_call, use_container_width=True)
            
        with col2:
            st.plotly_chart(fig_put, use_container_width=True)
            
        # Display key values
        st.markdown("### Key Values")
        sigma_decimal = sigma / 100
        r_decimal = r / 100
        q_decimal = q / 100
        
        bs_call = callBS(S, K, T, sigma_decimal, r_decimal, q_decimal)
        bs_put = putBS(S, K, T, sigma_decimal, r_decimal, q_decimal)
        
        bin_call = callEuropean(S, K, T, sigma_decimal, r_decimal, q_decimal, maxN * 2)
        bin_put = putEuropean(S, K, T, sigma_decimal, r_decimal, q_decimal, maxN * 2)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Call Options:**")
            st.write(f"Black-Scholes Value: ${bs_call:.4f}")
            st.write(f"Binomial Value (N={maxN*2}): ${bin_call:.4f}")
            st.write(f"Difference: ${abs(bs_call - bin_call):.4f}")
            
        with col2:
            st.write("**Put Options:**")
            st.write(f"Black-Scholes Value: ${bs_put:.4f}")
            st.write(f"Binomial Value (N={maxN*2}): ${bin_put:.4f}")
            st.write(f"Difference: ${abs(bs_put - bin_put):.4f}")
            
    except Exception as e:
        st.error(f"Error generating plots: {str(e)}")
        st.write("Please check your input parameters and try again.")