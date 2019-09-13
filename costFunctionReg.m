function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
su1 = 0;
su2 = 0;
su3 = 0;


  a = X * theta;
  h = sigmoid(a);
  
  for i = 1 : m
    
    su1 = su1 + (- (y(i) * log(h(i))) - ((1 - y(i)) * log( 1 - h(i))));
    
  endfor
  
  su2 = su1/m;
  
  
  for k = 2 : size(theta)
    
    su3 = su3 + theta(k)^2;
    
  endfor
  
  su2 = su2 + (lambda * su3/(2 * m));
  J = su2;
  
  
  
  for i = 1 : size(theta)
    
    
    
    if( i != 1)
    
      grad(i) = sum((h - y) .* X(:, i))/m  + (lambda * theta(i))/m;
      
    else
      
      grad(i) = sum((h - y) .* X(:, i))/m;
    
    endif  
    
  endfor

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
