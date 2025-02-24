# Black-Scholes Options Calculator

<div align="center">
  
![Python](https://img.shields.io/badge/Python-3.7+-green)

</div>

## Overview

A simple options pricer implementing the Black-Scholes model for European-style options. This calculator provides accurate theoretical prices and risk metrics for call and put options across multiple international markets.

## Key Features

- **Comprehensive Options Pricing** - Calculate theoretical prices for European call and put options
- **Multi-Market Support** - Pricing for US, UK, and European markets with appropriate currency formatting
- **Position Analysis** - Visualize payoff and PnL profiles for both long and short positions
- **Greeks Calculation:**
  - Delta - Sensitivity to underlying price changes
  - Gamma - Rate of change of Delta
  - Theta - Time decay effect
  - Vega - Sensitivity to volatility changes
  - Rho - Sensitivity to interest rate changes
- **Visualization** - Plots showing payoff profiles and Greeks evolution

## Supported Markets

| Region | Exchanges | Currency |
|--------|-----------|----------|
| North America | US Markets | $ |
| Europe | Euronext Paris, Amsterdam, Brussels, Lisbon, Deutsche Börse, Borsa Italiana, Bolsa de Madrid | € |
| United Kingdom | London Stock Exchange | £ |

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/black-scholes-calculator.git
cd black-scholes-calculator

# Install required packages
pip install yfinance numpy scipy matplotlib
```

## Usage

Running the calculator is straightforward:

```bash
python black_scholes_model.py
```

Follow the interactive prompts to:
1. Select a market
2. Enter a stock symbol
3. Specify expiration time in months
4. Set the strike price
5. Choose between long or short position
6. View optional Greeks evolution graphs

## Model Details

The Black-Scholes model calculates option prices using five key inputs:
- Current stock price (S)
- Option strike price (K)
- Time to expiration (T)
- Risk-free interest rate (r)
- Stock price volatility (σ)

## Limitations

- Assumes European-style options (exercise at expiration only)
- Uses historical volatility rather than implied volatility
- Does not account for dividends
- Uses approximate risk-free rates rather than real-time market data
- Assumes log-normal distribution of stock prices

