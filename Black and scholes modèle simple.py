import yfinance as yf
import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt

# Exchange dictionary
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
    # BlackScholes Model
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

    # Greeks calculations
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

    def rho_call(self):
        return self.K * self.T * np.exp(-self.r * self.T) * si.norm.cdf(self.d2())

    def rho_put(self):
        return -self.K * self.T * np.exp(-self.r * self.T) * si.norm.cdf(-self.d2())

    # Payoff and PnL calculations
    def plot_option_profiles(self, currency, position_type="long"):
        """Plot payoff and PnL profiles for both call and put options

        Args:
            currency (str): Currency symbol for axis labels
            position_type (str): "long" for buying options or "short" for selling options
        """
        # Generate price range for x-axis
        price_range = np.linspace(self.S * 0.5, self.S * 1.5, 100)

        # Calculate initial option prices
        call_price = self.call_option_price()
        put_price = self.put_option_price()

        # Calculate payoffs and PnL based on position
        if position_type.lower() == "long":
            # Long positions
            call_payoff = np.maximum(price_range - self.K, 0)
            call_pnl = call_payoff - call_price

            put_payoff = np.maximum(self.K - price_range, 0)
            put_pnl = put_payoff - put_price

            position_label = "Long"
        else:
            # Short positions
            call_payoff = np.minimum(self.K - price_range, 0)
            call_pnl = call_price + call_payoff

            put_payoff = np.minimum(price_range - self.K, 0)
            put_pnl = put_price + put_payoff

            position_label = "Short"

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot Call Option
        ax1.plot(price_range, call_payoff, 'b-', label='Payoff')
        ax1.plot(price_range, call_pnl, 'r--', label='PnL')
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax1.axvline(x=self.K, color='k', linestyle='--', alpha=0.3)
        ax1.set_title(f'{position_label} Call Option Profile')
        ax1.set_xlabel(f'Stock Price ({currency})')
        ax1.set_ylabel(f'Profit/Loss ({currency})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot Put Option
        ax2.plot(price_range, put_payoff, 'b-', label='Payoff')
        ax2.plot(price_range, put_pnl, 'r--', label='PnL')
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.axvline(x=self.K, color='k', linestyle='--', alpha=0.3)
        ax2.set_title(f'{position_label} Put Option Profile')
        ax2.set_xlabel(f'Stock Price ({currency})')
        ax2.set_ylabel(f'Profit/Loss ({currency})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# Plotting Greeks evolution
def plot_greeks_evolution(bs_model, currency):
    """
    Plot the evolution of Greeks with respect to their associated parameters

    Args:
        bs_model: BlackScholesModel instance with current parameters
        currency: Currency symbol for axis labels
    """
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Stock price range
    stock_prices = np.linspace(bs_model.S * 0.5, bs_model.S * 1.5, 100)

    # Time to expiration range
    times = np.linspace(1 / 120, bs_model.T, 100)

    # Volatility range
    sigmas = np.linspace(0.1, 1.0, 100)

    # Interest rate range
    rates = np.linspace(0.001, 0.10, 100)

    # 1. Delta vs Stock Price
    call_deltas = []
    put_deltas = []
    for s in stock_prices:
        temp_model = BlackScholesModel(s, bs_model.K, bs_model.T, bs_model.r, bs_model.sigma)
        call_deltas.append(temp_model.delta_call())
        put_deltas.append(temp_model.delta_put())

    axes[0, 0].plot(stock_prices, call_deltas, 'b-', label='Call Delta')
    axes[0, 0].plot(stock_prices, put_deltas, 'r--', label='Put Delta')
    axes[0, 0].axvline(x=bs_model.K, color='k', linestyle='--', alpha=0.3)
    axes[0, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 0].set_title('Delta vs Stock Price')
    axes[0, 0].set_xlabel(f'Stock Price ({currency})')
    axes[0, 0].set_ylabel('Delta')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Gamma vs Stock Price
    gammas = []
    for s in stock_prices:
        temp_model = BlackScholesModel(s, bs_model.K, bs_model.T, bs_model.r, bs_model.sigma)
        gammas.append(temp_model.gamma())

    axes[0, 1].plot(stock_prices, gammas, 'g-')
    axes[0, 1].axvline(x=bs_model.K, color='k', linestyle='--', alpha=0.3)
    axes[0, 1].set_title('Gamma vs Stock Price')
    axes[0, 1].set_xlabel(f'Stock Price ({currency})')
    axes[0, 1].set_ylabel('Gamma')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Theta vs Time to Expiration
    call_thetas = []
    put_thetas = []
    for t in times:
        temp_model = BlackScholesModel(bs_model.S, bs_model.K, t, bs_model.r, bs_model.sigma)
        call_thetas.append(temp_model.theta_call())
        put_thetas.append(temp_model.theta_put())

    axes[0, 2].plot(times * 365, call_thetas, 'b-', label='Call Theta')  # Convert to days
    axes[0, 2].plot(times * 365, put_thetas, 'r--', label='Put Theta')
    axes[0, 2].set_title('Theta vs Days to Expiration')
    axes[0, 2].set_xlabel('Days to Expiration')
    axes[0, 2].set_ylabel(f'Theta ({currency}/day)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Vega vs Volatility
    vegas = []
    for sigma in sigmas:
        temp_model = BlackScholesModel(bs_model.S, bs_model.K, bs_model.T, bs_model.r, sigma)
        vegas.append(temp_model.vega())

    axes[1, 0].plot(sigmas * 100, vegas, 'm-')  # Convert to percentage
    axes[1, 0].axvline(x=bs_model.sigma * 100, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].set_title('Vega vs Volatility')
    axes[1, 0].set_xlabel('Volatility (%)')
    axes[1, 0].set_ylabel(f'Vega ({currency})')
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Rho vs Interest Rate
    call_rhos = []
    put_rhos = []
    for r in rates:
        temp_model = BlackScholesModel(bs_model.S, bs_model.K, bs_model.T, r, bs_model.sigma)
        call_rhos.append(temp_model.rho_call())
        put_rhos.append(temp_model.rho_put())

    axes[1, 1].plot(rates * 100, call_rhos, 'b-', label='Call Rho')  # Convert to percentage
    axes[1, 1].plot(rates * 100, put_rhos, 'r--', label='Put Rho')
    axes[1, 1].axvline(x=bs_model.r * 100, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].set_title('Rho vs Interest Rate')
    axes[1, 1].set_xlabel('Interest Rate (%)')
    axes[1, 1].set_ylabel(f'Rho ({currency})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 6. All Greeks vs Time to Expiration (normalized)
    deltas_t = []
    gammas_t = []
    thetas_t = []
    vegas_t = []
    rhos_t = []

    for t in times:
        temp_model = BlackScholesModel(bs_model.S, bs_model.K, t, bs_model.r, bs_model.sigma)
        deltas_t.append(temp_model.delta_call())
        gammas_t.append(temp_model.gamma())
        thetas_t.append(temp_model.theta_call() / max(abs(np.min(call_thetas)), abs(np.max(call_thetas))))
        vegas_t.append(temp_model.vega() / max(vegas))
        rhos_t.append(temp_model.rho_call() / max(abs(np.min(call_rhos)), abs(np.max(call_rhos))))

    axes[1, 2].plot(times * 365, deltas_t, 'b-', label='Delta')
    axes[1, 2].plot(times * 365, gammas_t, 'g-', label='Gamma')
    axes[1, 2].plot(times * 365, thetas_t, 'r-', label='Theta')
    axes[1, 2].plot(times * 365, vegas_t, 'm-', label='Vega')
    axes[1, 2].plot(times * 365, rhos_t, 'c-', label='Rho')
    axes[1, 2].set_title('Normalized Greeks vs Days to Expiration')
    axes[1, 2].set_xlabel('Days to Expiration')
    axes[1, 2].set_ylabel('Normalized Value')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Helper functions
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
    """Approximate risk-free rate"""
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
    return rates.get(exchange, 0.04)


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
        print(f"Call Rho: {bs.rho_call():.4f}")
        print(f"Put Rho: {bs.rho_put():.4f}")

        # Ask if user wants to go long or short
        position_type = ""
        while position_type not in ["long", "short"]:
            position_type = input("\nDo you want to go long or short on the options? (long/short): ").lower()
            if position_type not in ["long", "short"]:
                print("Please enter either 'long' or 'short'.")

        # Plot payoff and PnL diagrams
        bs.plot_option_profiles(currency, position_type)

        # Ask if user wants to see the Greeks evolution
        see_greeks = input("\nDo you want to see the Greeks evolution graphs? (yes/no): ").lower()
        if see_greeks.startswith('y'):
            plot_greeks_evolution(bs, currency)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
