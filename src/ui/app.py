import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import requests
from io import BytesIO
import random

# Import your custom modules
from DDPGPortfolioRecommender import DDPGPortfolioRecommender
from utils.recommender_utils import (
    simulate_future_recommendations_with_realistic_profits,
    display_date_based_recommendations,
    export_date_based_recommendations,
    plot_trading_summary,
)

# Add methods to the class
DDPGPortfolioRecommender.simulate_future_recommendations_with_realistic_profits = (
    simulate_future_recommendations_with_realistic_profits
)
DDPGPortfolioRecommender.display_date_based_recommendations = (
    display_date_based_recommendations
)
DDPGPortfolioRecommender.export_date_based_recommendations = (
    export_date_based_recommendations
)
DDPGPortfolioRecommender.plot_trading_summary = (
    plot_trading_summary
)

# Function to download files from HuggingFace
def download_from_hf(repo_id, filename, local_dir="data"):
    # Create the directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    # Path to save the file
    local_path = os.path.join(local_dir, filename)
    
    # Only download if the file doesn't exist locally
    if not os.path.exists(local_path):
        try:
            # URL for the file on HuggingFace
            url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
            
            # Download the file
            response = requests.get(url)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            # Save the file
            with open(local_path, "wb") as f:
                f.write(response.content)
                
            st.success(f"Downloaded {filename} from HuggingFace")
        except Exception as e:
            st.error(f"Error downloading {filename}: {e}")
            return None
    
    return local_path

def main():
    st.title("DDPG Portfolio Recommender")
    
    # Replace with your HuggingFace dataset repo ID
    hf_repo_id = "tarunsamuel7/capstone_dataset"  # Update this!
    
    # Download files from HuggingFace if needed
    with st.spinner("Checking for model and data files..."):
        model_path = download_from_hf(hf_repo_id, "ddpg_portfolio_model.zip")
        data_path = download_from_hf(hf_repo_id, "historical_data.csv")
    
    if not model_path or not data_path:
        st.error("Failed to load required files. Please check the repository ID and file names.")
        return
    
    # User input for investment amount
    st.header("Investment Settings")
    investment_amount = st.number_input(
        "Investment Amount ($)",
        min_value=1000,
        max_value=100000,
        value=10000,
        step=1000,
    )
    
    # Button to run the simulation
    if st.button("Run Portfolio Simulation"):
        with st.spinner("Running simulation..."):
            try:
                # Create the recommender
                recommender = DDPGPortfolioRecommender(
                    model_path=model_path,
                    data_path=data_path,
                    max_stocks=100,
                    lookback=30,
                    feature_dimension=7,  # Adjust based on your model
                )
                
                # Run the simulation
                # date_recommendations, final_value, realized_profit = (
                #     recommender.simulate_future_recommendations_with_realistic_profits(
                #         amount_cad=investment_amount,
                #         future_days=7,
                #         price_change_range=(-0.15, 0.25),
                #     )
                # )
                
                # # Display the final value (as requested)
                # st.success("Simulation completed successfully!")
                # st.header("Results")
                # st.metric(
                #     label="Final Portfolio Value",
                #     value=f"${final_value:.2f}",
                #     delta=f"{(final_value/investment_amount - 1)*100:.2f}%"
                # )
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.error("Please check that your model and data paths are correct.")

if __name__ == "__main__":
    main()