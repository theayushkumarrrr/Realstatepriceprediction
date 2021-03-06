function [X_norm, mu, sigma] = featureNormalize(X)
%For feature scaling
mu = mean(X);
sigma = std(X);

t = ones(length(X), 1);
X_norm = (X - (t * mu)) ./ (t * sigma); 

end
