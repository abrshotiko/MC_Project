# Import necessary libraries
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from scipy.stats import norm, kstest
from scipy.stats import gaussian_kde  # Import for KDE

# Set up the Streamlit app layout
st.set_page_config(layout="wide")  # Wide layout to mimic the image design

# Title
st.title("Monte Carlo European Option Pricing")

# Add the "Created by" section in the sidebar using HTML
st.sidebar.markdown("""
    <div style="padding: 10px; border-radius: 5px; background-color: #1c1e21;">
        <h3 style="margin-bottom: 15px; font-family: 'Helvetica', sans-serif; color: white;">
            Monte Carlo Model  🚀
        </h3>
        <div style="display: flex; align-items: center;">
            <a href="https://www.linkedin.com/in/shotikoab/" target="_blank" style="display: flex; align-items: center; text-decoration: none;">
                <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="20" height="20" style="margin-right: 10px;">
                <p style="color: white; font-family: 'Helvetica', sans-serif; margin: 0;">Shotiko Abramishvili</p>
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)
# Sidebar for Inputs
st.sidebar.header("Monte Carlo Simulation Inputs")

# Sidebar Inputs
ticker = st.sidebar.text_input("Enter the stock ticker (e.g., AAPL for Apple):")
start_date = st.sidebar.date_input("Select start date for Stock Price Data:")
end_date = st.sidebar.date_input("Select end date for Stock Price Data:")

# Recommendation: Ensure that the date range is more than 1 year
if (end_date - start_date).days < 365:
    st.sidebar.warning("It's recommended to pick a more than 2 years for robustness")

# Monte Carlo parameters in Sidebar
num_simulations = st.sidebar.number_input("Number of simulations (minimum 10,000):", min_value=10000, max_value=100000, value=10000)
TTM = st.sidebar.number_input("Time to maturity (in years):", min_value=0.1, max_value=10.0, value=1.0)

# Option pricing parameters in Sidebar
strike_price = st.sidebar.number_input("Enter the strike price:", min_value=0.0, value=100.0)

# Button to trigger simulation
run_simulation = st.sidebar.button("Run Simulation")
# Sidebar disclaimer
st.sidebar.markdown("""
*Disclaimer: The 10-Year T-Bill is used as the risk-free rate for this simulation.
""")

# Main Output Area
if run_simulation:
    # Fetch stock data from Yahoo Finance
    df = yf.download(ticker, start=start_date, end=end_date)

    if not df.empty:
        st.success(f"Successfully fetched data for {ticker} from {start_date} to {end_date}.")
        # Display a sample of the stock data (first 5 rows)
        st.subheader("Sample Data")
        st.dataframe(df.head(), use_container_width=True)  # Make the DataFrame use the full width

        # Expandable box for the full data
        with st.expander("See full stock data"):
            st.write("Full stock data:")
            st.dataframe(df)  # Show the full dataset inside the expander

        # Calculate log returns
        df['Log Returns'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
        df = df.dropna()  # Drop rows with NaN values caused by shift

        # Assign log returns to the numpy array variable `log_returns`
        log_returns = df['Log Returns'].to_numpy()

        # Fix the date format on x-axis for better display (reduce number of dates)
        df['Date'] = df.index
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

        # Fit a normal distribution to the log returns
        mu, sigma = norm.fit(log_returns)

        # Test for normality using the Kolmogorov-Smirnov test
        ks_statistic, p_value = kstest(log_returns, 'norm', args=(mu, sigma))

        # Display the results of the normality test in a single box
        result = "The log returns follow a normal distribution (fail to reject the null hypothesis)." if p_value > 0.05 else "The log returns do not follow a normal distribution (reject the null hypothesis)."

        # Use HTML to format the result nicely in a single box with some fancy layout
        st.markdown(f"""
            <div style="padding: 20px; background-color: #2b2b2b; border-radius: 10px; text-align: center; color: white;">
                <h3 style="margin: 0; font-size: 24px;">Normality Test Results</h3>
                <div style="display: flex; justify-content: space-around; margin-top: 10px;">
                    <div>
                        <strong>Kolmogorov-Smirnov Statistic:</strong> {ks_statistic:.6f}
                    </div>
                    <div>
                        <strong>P-value:</strong> {p_value:.6f}
                    </div>
                </div>
                <div style="margin-top: 20px; padding: 10px; border-radius: 5px; background-color: {'#90ee90' if p_value > 0.05 else '#ffcccb'};">
                    <strong>{result}</strong>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Create dynamic columns for graphs with equal size and fixed dimensions
        col1, col2, col3 = st.columns([1, 1, 1])

        # Log Returns Graph (Column 1)
        with col1:
            st.subheader(f"Log Returns for {ticker}")
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(df['Date'], df['Log Returns'], label='Log Returns', color='blue')
            ax.set_title(f"Log Returns for {ticker}")
            ax.set_xlabel('Date')
            ax.set_ylabel('Log Returns')
            
            # Adjust date ticks to display fewer labels
            ax.set_xticks(np.arange(0, len(df['Date']), len(df['Date']) // 10))  # Show 10 date ticks only
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True)
            st.pyplot(fig)

        # Normal Distribution Fit (Column 2)
        with col2:
            st.subheader(f"Fitting Normal Distribution")
            fig, ax = plt.subplots(figsize=(5, 4.27))
            ax.hist(log_returns, bins=50, density=True, alpha=0.6, color='green', label='Log Returns Histogram')
            x = np.linspace(min(log_returns), max(log_returns), 100)
            p = norm.pdf(x, mu, sigma)
            ax.plot(x, p, 'black', linewidth=2, label='Fitted Normal Distribution')
            ax.set_title(f"Normal Distribution Fit")
            ax.legend()
            st.pyplot(fig)

        # If the normal distribution does not hold, display KDE
        if p_value <= 0.05:
            kde = gaussian_kde(log_returns)

            # Empirical PDF Using KDE (Column 3)
            with col3:
                st.subheader("Empirical PDF Using KDE")
                fig, ax = plt.subplots(figsize=(5, 4.27))
                ax.hist(log_returns, bins=50, density=True, alpha=0.6, color='green', label='Log Returns Histogram')
                PDF_empirical = kde(x)
                ax.plot(x, PDF_empirical, 'red', linewidth=2, label='KDE Empirical PDF')
                ax.set_title(f"Empirical PDF (KDE)")
                ax.legend()
                st.pyplot(fig)

            # Use KDE sigma for Monte Carlo simulation
            sigma_mc = np.std(kde.resample(size=1000))  # Standard deviation from KDE sample
            mu_mc = np.mean(log_returns)  # Empirical mean of log returns for Monte Carlo
        else:
            sigma_mc = sigma  # Use normal distribution sigma for Monte Carlo simulation
            mu_mc = mu        # Use normal distribution mu for Monte Carlo simulation

        # Fetch the 10-year US Treasury yield from Yahoo Finance
        treasury_yield_data = yf.download('^TNX', period='1d', interval='1d')
        if not treasury_yield_data.empty:
            risk_free_rate = treasury_yield_data['Close'].iloc[-1] / 100  # Convert percentage to decimal
        else:
            st.error("Could not fetch the 10-year US Treasury yield.")
            risk_free_rate = 0.035  # Fallback rate

        # Use the risk-free rate as mu for the stock
        mu_mc = risk_free_rate

        # Align the Monte Carlo graph and option prices side by side
        colA, colB = st.columns([2, 1])  # 2 parts for the MC graph, 1 part for the options prices

        with colA:
            # Run Monte Carlo simulation using the drift mu and volatility sigma from the fitted distribution
            st.subheader("Running Monte Carlo Simulation")

            S0 = df['Adj Close'].iloc[-1]  # Starting price (current stock price)
            dt = 1 / 252  # Daily time step (assuming 252 trading days in a year)
            time_steps = int(TTM * 252)  # Convert time to maturity into an integer number of steps
            price_paths = np.zeros((time_steps, num_simulations))
            price_paths[0] = S0

            cmap = get_cmap('tab20')  # Using a colormap for different simulation colors
            fig, ax = plt.subplots(figsize=(10, 6))

            for t in range(1, time_steps):
                Z = np.random.standard_normal(num_simulations)
                price_paths[t] = price_paths[t - 1] * np.exp((mu_mc - 0.5 * sigma_mc**2) * dt + sigma_mc * np.sqrt(dt) * Z)

            for i in range(min(num_simulations, 100)):  # Display up to 100 simulation paths
                ax.plot(price_paths[:, i], color=cmap(i % 20), lw=0.7, alpha=0.7)

            ax.set_title(f"Simulated Stock Price Paths ({num_simulations} simulations)", fontsize=16, color='darkblue')
            ax.set_xlabel("Days", fontsize=12)
            ax.set_ylabel("Stock Price", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)

        with colB:
            # Option pricing based on simulated price paths
            st.subheader(f"Strike Price {strike_price}")

            # Calculate payoffs for Call and Put
            call_payoffs = np.maximum(price_paths[-1] - strike_price, 0)
            put_payoffs = np.maximum(strike_price - price_paths[-1], 0)

            # Discounting the expected payoff using the risk-free rate
            call_option_price = np.exp(-risk_free_rate * TTM) * np.mean(call_payoffs)
            put_option_price = np.exp(-risk_free_rate * TTM) * np.mean(put_payoffs)

            # Use fixed height for both boxes for alignment
            st.markdown(f"""
                <div style="height: 270px; background-color: #90ee90; display: flex; flex-direction: column; justify-content: center; align-items: center; border-radius: 5px; text-align: center; margin-bottom: 20px;">
                    <h2 style="margin: 0;">Call Option Price</h2>
                    <h2 style="margin: 0;">${call_option_price:.2f}</h2>
                </div>
                <div style="height: 270px; background-color: #ffcccb; display: flex; flex-direction: column; justify-content: center; align-items: center; border-radius: 5px; text-align: center;">
                    <h2 style="margin: 0;">Put Option Price</h2>
                    <h2 style="margin: 0;">${put_option_price:.2f}</h2>
                </div>
            """, unsafe_allow_html=True)

    else:
        st.error("No data found for the given dates. Please check the dates and ticker.")
