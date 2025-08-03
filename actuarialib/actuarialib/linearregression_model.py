class LinearRegression:
    def __init__(self):
        self.alpha = 0
        self.beta = 0
        self.sigmasquare = 0
        self.fitted = False

    def fit(self, x, y):
        n = len(x)
        x_bar = sum(x) / n
        y_bar = sum(y) / n

        sxx = sum((xi - x_bar)**2 for xi in x)
        syy = sum((yi - y_bar)**2 for yi in y)
        sxy = sum((x[i] - x_bar) * (y[i] - y_bar) for i in range(n))

        self.beta = sxy / sxx
        self.alpha = y_bar - self.beta * x_bar
        self.Sxx, self.Syy, self.Sxy, self.n = sxx, syy, sxy, n
        self.fitted = True

    def variance(self):
        if not self.fitted:
            raise ValueError("Model must be fitted before calculating variance.")
        self.sigmasquare = (self.Syy - (self.Sxy ** 2) / self.Sxx) / (self.n - 2)
        return self.sigmasquare

    def predict(self, x_values):
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction.")
        if isinstance(x_values, (int, float)):
            return self.alpha + self.beta * x_values
        return [self.alpha + self.beta * x for x in x_values]
