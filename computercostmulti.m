function J = computeCostMulti(X, y, theta)
% Cost Function

m = length(y); 

J = (1/(2*m)) * (X * theta - y)' * (X * theta - y); 

end