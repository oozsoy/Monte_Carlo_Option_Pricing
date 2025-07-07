
import numpy as np
import matplotlib.pyplot as plt

class MonteCarloSimulator:
    
    def __init__(self, S0, T, r, n_paths, n_steps, model_type='GBM', model_params=None):
        self.S0 = S0
        self.T = T
        self.r = r
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.model_type = model_type
        self.dt = T / n_steps
        self.model_params = model_params or {}
        self.paths = None
        
    
    def simulate_paths(self):
        
        if self.model_type == 'GBM':
            return self._simulate_gbm()
        elif self.model_type == 'Heston':
            return self._simulate_heston()
        else:
            raise NotImplementedError(f"{self.model_type} is not supported")

    def _simulate_gbm(self):
        
        sigma = self.model_params.get('sigma', 0.2)
        dt = self.dt
        paths = np.zeros((self.n_paths, self.n_steps + 1))
        paths[:, 0] = self.S0
        
        for t in range(1, self.n_steps + 1):
            z = np.random.normal(size=self.n_paths)
            paths[:, t] = paths[:, t-1] * np.exp((self.r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
            
        self.paths = paths
        return paths
    
    def _simulate_heston(self, return_var=False):
        """
        Simulate asset price paths using the Heston stochastic volatility model.
        Returns paths with shape (n_paths, n_steps + 1), consistent with GBM.
        """
        # Extract Heston parameters
        v0 = self.model_params.get('v0', 0.04)       # Initial variance
        kappa = self.model_params.get('kappa', 1.5)  # Mean reversion speed
        theta = self.model_params.get('theta', 0.04) # Long-run average variance
        xi = self.model_params.get('xi', 0.5)        # Vol of vol
        rho = self.model_params.get('rho', -0.7)

        dt = self.dt
        N, M = self.n_steps, self.n_paths
    
        # Initialize paths
        S = np.zeros((M, N + 1))
        v = np.zeros((M, N + 1))
        
        S[:, 0] = self.S0
        v[:, 0] = v0

        # Correlated Brownian increments
        #Z = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=(M, N))  # shape: (M, N, 2)

        for i in range(1, N + 1):
            
            
            Z = np.random.multivariate_normal(np.array([0,0]), cov = np.array([[1,rho],[rho,1]]), size=M) 
            #v_prev = np.maximum(v_prev, 1e-8)
            v_prev = v[:, i - 1]

            sqrt_v_dt = np.sqrt(v_prev * dt)

            v[:, i] = np.maximum(v_prev + kappa * (theta - v_prev) * dt + xi * sqrt_v_dt * Z[:,1], 0)
            S[:, i] = S[:, i - 1] * np.exp((self.r - 0.5 * v_prev) * dt + sqrt_v_dt * Z[:,0])

        self.paths = S
        self.variance_paths = v

        return (S, v) if return_var else S

    
    def price_option(self, payoff_fn):
        if self.paths is None:
            
            self.simulate_paths()
            
            payoffs = payoff_fn(self.paths[:, -1])
            discounted = np.exp(-self.r * self.T) * np.mean(payoffs)
        
        return discounted
    
    def plot_paths(self, n=10):
        
        if self.paths is None:
            self.simulate_paths()
            
        t = np.linspace(0, self.T, self.n_steps + 1)
        
        plt.figure(figsize = (9,5))
        
        for i in range(min(n, self.n_paths)):
            
            plt.plot(t, self.paths[i], color = 'k', alpha = 0.3)
        
        plt.title(f"Sample Simulated Paths ({self.model_type})")
        plt.xlabel("Time [years]")
        plt.ylabel("Asset Price")
        plt.grid(alpha = 0.3)
        plt.show()