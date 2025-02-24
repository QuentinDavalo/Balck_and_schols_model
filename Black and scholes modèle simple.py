import yfinance as yf
import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt

# Previous exchange dictionary remains the same
EXCHANGES = {
    'US': {'suffix': '', 'currency': '$', 'name': 'US Markets'},
    'PA': {'suffix': '.PA', 'currency': '€', 'name': 'Euronext Paris'},
    'AS': {'suffix': '.AS', 'currency': '€', 'name': 'Euronext Amsterdam'},
    'BR': {'suffix': '.BR', 'currency': '€', 'name': 'Euronext Brussels'},
    'LI': {'suffix': '.LI', 'currency': '€', 'name': 'Euronext Lisbon'},
    'LSE': {'suffix': '.L', 'currency': '£', 'name': 'London Stock Exchange'},
    'DE': {'suffix': '.DE', 'currency': '€', 'name': 'Deutsche Börse'},
    'MI': {'suffix': '.MI', 'currency': '€', 'name': 'Borsa Italiana'},
    'MC': {'suffix': '.MC', 'currency': '€', 'name': 'Bolsa de Madrid'},
}


class BlackScholesModel:
    # Previous BlackScholesModel class methods remain the same
    def __init__(self, S, K, T, r, sigma):
        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)

    def d1(self):
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))

    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)

    def call_option_price(self):
        if self.T <= 0:
            return max(0, self.S - self.K)
        return (self.S * si.norm.cdf(self.d1()) -
                self.K * np.exp(-self.r * self.T) * si.norm.cdf(self.d2()))

    def put_option_price(self):
        if self.T <= 0:
            return max(0, self.K - self.S)
        return (self.K * np.exp(-self.r * self.T) * si.norm.cdf(-self.d2()) -
                self.S * si.norm.cdf(-self.d1()))

    # Previous Greeks calculations remain the same
    def delta_call(self):
        return si.norm.cdf(self.d1())

    def delta_put(self):
        return -si.norm.cdf(-self.d1())

    def gamma(self):
        return si.norm.pdf(self.d1()) / (self.S * self.sigma * np.sqrt(self.T))

    def theta_call(self):
        return (-self.S * si.norm.pdf(self.d1()) * self.sigma / (2 * np.sqrt(self.T)) -
                self.r * self.K * np.exp(-self.r * self.T) * si.norm.cdf(self.d2()))

    def theta_put(self):
        return (-self.S * si.norm.pdf(self.d1()) * self.sigma / (2 * np.sqrt(self.T)) +
                self.r * self.K * np.exp(-self.r * self.T) * si.norm.cdf(-self.d2()))

    def vega(self):
        return self.S * si.norm.pdf(self.d1()) * np.sqrt(self.T)

    # New methods for payoff and PnL calculations
    def plot_option_profiles(self, currency):
        """Plot payoff and PnL profiles for both call and put options"""
        # Generate price range for x-axis (from 50% to 150% of current price)
        price_range = np.linspace(self.S * 0.5, self.S * 1.5, 100)

        # Calculate initial option prices
        call_price = self.call_option_price()
        put_price = self.put_option_price()

        # Calculate payoffs and PnL for call option
        call_payoff = np.maximum(price_range - self.K, 0)
        call_pnl = call_payoff - call_price

        # Calculate payoffs and PnL for put option
        put_payoff = np.maximum(self.K - price_range, 0)
        put_pnl = put_payoff - put_price

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot Call Option
        ax1.plot(price_range, call_payoff, 'b-', label='Payoff')
        ax1.plot(price_range, call_pnl, 'r--', label='PnL')
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax1.axvline(x=self.K, color='k', linestyle='--', alpha=0.3)
        ax1.set_title('Call Option Profile')
        ax1.set_xlabel(f'Stock Price ({currency})')
        ax1.set_ylabel(f'Profit/Loss ({currency})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot Put Option
        ax2.plot(price_range, put_payoff, 'b-', label='Payoff')
        ax2.plot(price_range, put_pnl, 'r--', label='PnL')
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.axvline(x=self.K, color='k', linestyle='--', alpha=0.3)
        ax2.set_title('Put Option Profile')
        ax2.set_xlabel(f'Stock Price ({currency})')
        ax2.set_ylabel(f'Profit/Loss ({currency})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# Previous helper functions remain the same
def select_exchange():
    """Let user select the stock exchange"""
    print("\nAvailable Markets:")
    for code, info in EXCHANGES.items():
        print(f"{code}: {info['name']}")

    while True:
        exchange = input("\nSelect market code (e.g., US, PA, LSE): ").upper()
        if exchange in EXCHANGES:
            return exchange
        print("Invalid market code. Please try again.")


def get_stock_data(ticker_symbol, exchange):
    """Fetches historical stock data with exchange suffix"""
    full_symbol = ticker_symbol + EXCHANGES[exchange]['suffix']
    stock = yf.Ticker(full_symbol)
    data = stock.history(period='1y')
    if len(data) == 0:
        raise ValueError(f"No data found for {full_symbol}. Please check the ticker symbol.")
    return data


def calculate_historical_volatility(stock_data, window=252):
    """Calculates historical volatility"""
    log_returns = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
    return np.sqrt(window) * log_returns.std()


def get_risk_free_rate(exchange):
    """Returns an approximate risk-free rate based on the market"""
    rates = {
        'US': 0.05,
        'PA': 0.03,
        'AS': 0.03,
        'BR': 0.03,
        'LI': 0.03,
        'LSE': 0.04,
        'DE': 0.03,
        'MI': 0.03,
        'MC': 0.03
    }
    return rates.get(exchange, 0.03)


def main():
    try:
        # Select exchange
        exchange = select_exchange()
        currency = EXCHANGES[exchange]['currency']

        # Get stock symbol
        ticker_symbol = input(f"\nEnter the stock symbol for {EXCHANGES[exchange]['name']}: ")
        months = float(input("Enter the time to expiration in months: "))

        # Get stock data
        stock_data = get_stock_data(ticker_symbol, exchange)
        S = stock_data['Close'].iloc[-1]

        print(f"\nCurrent price: {currency}{S:.2f}")
        K = float(input(f"Enter the strike price ({currency}): "))

        # Calculate parameters
        volatility = calculate_historical_volatility(stock_data)
        r = get_risk_free_rate(exchange)
        T = months / 12

        # Calculate option prices and greeks
        bs = BlackScholesModel(S, K, T, r, volatility)

        # Display results
        print("\nResults:")
        print("-" * 50)
        print(f"Historical Volatility: {volatility:.2%}")
        print(f"Risk-free Rate: {r:.2%}")
        print(f"\nOption Prices:")
        print(f"Call: {currency}{bs.call_option_price():.2f}")
        print(f"Put: {currency}{bs.put_option_price():.2f}")
        print(f"\nGreeks:")
        print(f"Call Delta: {bs.delta_call():.4f}")
        print(f"Put Delta: {bs.delta_put():.4f}")
        print(f"Gamma: {bs.gamma():.4f}")
        print(f"Call Theta: {currency}{bs.theta_call():.4f}")
        print(f"Put Theta: {currency}{bs.theta_put():.4f}")
        print(f"Vega: {currency}{bs.vega():.4f}")

        # Plot payoff and PnL diagrams
        bs.plot_option_profiles(currency)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()