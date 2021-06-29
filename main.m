clear ; close all; clc

fprintf('Loading data ...\n');

%% Load the data set into the system
data = load('dataset.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Print the first 10 data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

% Feature scaling
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X)


X = [ones(m, 1) X];


%Running gradient descent
fprintf('Running gradient descent ...\n');

% learning rate
alpha = 0.01;
num_iters = 1000;
 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);


prices_x=X(:,1);
brs_x=X(:,2);


%Plot the data set
figure;

plot3(prices_x,brs_x,y);

xlabel('Area of houses');
ylabel('No of bedrooms');
zlabel('Price of house');


% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
%plot(1:50, J1(1:50), 'b');
hold on;
%plot(1:50, J2(1:50), 'r');
%plot(1:50, J3(1:50), 'k');
xlabel('Number of iterations');
ylabel('Cost J');



plot(prices_x,brs_x,y);

fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

area_house= (1650-mu(1))/sigma(1);
br_house= (3-mu(2))/sigma(2);
price = theta' * [1;area_house;br_house]; 

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);