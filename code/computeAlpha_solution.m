% This is Exercise 2 of the course "Model Predictive Control" in the summer
% term taught at the IST.

function computeAlpha_solution


% In this exercise, a suitable terminal set for a
% Quasi-Infinite-Horizon MPC scheme shall be computed.
% 
% Here, we use the 4-step procedure described in the paper[ Quasi-Infinite Horizon
% Nonlinear Model Predictive Control Scheme with Guaranteed Stability] by
% Chen & Allgöwer. 
%
% It is recomended to use at leat version R2014b of MATLAB.
% 
% You will need the following two addons for MATLAB:
% - Multi-Parametric Toolbox 3 (MPT3):
%   http://control.ee.ethz.ch/~mpt/3/Main/Installation
% - ellipsoids - Ellipsoidal Toolbox 1.1.3 lite:
%   http://code.google.com/p/ellipsoids/downloads/list
% 
% ==========================================================
% 
% You only have to replace the ??? fields in the code!
% 
% ==========================================================

clear all,
close all,
clc
 
warning off

%% Preliminaries

% Jacobian linearization of the system
%A = [0 1; 1 0];
%B = [0.5; 0.5];
%Q = 0.5*eye(2);
%R = 1;
M = .5;
m = 0.2;
b = 0.1;
I = 0.006;
g = 9.8;
l = 0.3;

p = I*(M+m)+M*m*l^2; %denominator for the A and B matrices

A = [0      1              0           0;
     0 -(I+m*l^2)*b/p  (m^2*g*l^2)/p   0;
     0      0              0           1;
     0 -(m*l*b)/p       m*g*l*(M+m)/p  0];

B = [     0;
     (I+m*l^2)/p;
          0;
        m*l/p];


% dimensions (state and input)
n = length(A(1,:));
m = length(B(1,:));

Q = eye(n);
R = 0.01;

%% Step 1: Local controller (LQR)
disp('Step 1')

% Solution of the Algebraic Riccati Equation (ARE)
P_LQR = care(A,B,Q,R);

% LQR controller
K = -inv(R)*B.'*P_LQR

% closed-loop system 
AK = A+B*K; % Attention! The lqr command produces A-B*K, whereas in the paper A+B*K is used!
disp('eig(AK)')
eig(AK)

pause

disp('______________________________')

%% Step 2: Derive P
disp('Step 2')

% Choose kappa
kappa = 0.01
max(real(eig(AK)))
% check condition
if kappa >= -max(real(eig(AK)))
    error('kappa >= -max(real(eig(AK)))')
end

% solve Lyapunov equation
P=lyap((AK+kappa*eye(n))',Q+K'*R*K);
P
%%
pause

disp('______________________________')

%% Step 3: Derive alpha_1
disp('Step 3')
% We determine the largest alpha_l, such that the Kx\in U \forall x\in X_f
% This is done by looking at each constraint and obtaining the maximum alpha, such that the inequality constraint is fullfilled with equality.
% --> Then we take the smallest of the alpha values to be alpha_l
% Computationally more effient, this can also be posed as a linear program (LP) 
% (State Constraints can be included in a similar fashion.)

% input constraints (box constraints)
u_min = -200;
u_max =  200;

% analytical variables
uu = sdpvar(m,1);

% constraint sets (inequalities)
MU = [ uu >= u_min, uu <= u_max ];

% conversion into polyhedra
U = Polyhedron(MU);

% conversion of polyhedron into its minimal representations
U.minHRep();

clear('uu','MU')

% options for optimization
options = sdpsettings('solver','fmincon','verbose',0);

% number of input constraints
mu = length(U.A(:,1));

alpha_1_vec=[];

for k = 1:mu
    
    % definition of the optimization variable
    x_opt = sdpvar(n,1);
    
    % cost function
    H = x_opt' * P * x_opt;
    
    % constraints
    constraints = [U.A(k,:) * K * x_opt == U.b(k,:)];
    
    % solve optimization problem
    sol = optimize(constraints,H,options);
    alpha_1_vec = [alpha_1_vec, double(H)];
    
end
clear('k','constraints','x_opt','H','sol','mu','mx')

alpha_1 = min(alpha_1_vec)

pause

disp('______________________________')

%% Step 4: Find suitable alpha
disp('Step 4')
%In this step we search for a alpha<=alpha_l, such that
%L_Phi<=L_Phi_max.
%This is done with a bisection.

% upper bound for L_Phi
L_Phi_max = ( kappa * min(real(eig(P))) ) / (norm(P,2));

% initial conditions for optimization
alpha_ub = alpha_1;
alpha_lb = 0;
L_Phi = FcnL_phi(AK,K,P,alpha_1);
alpha = alpha_1;
exitflag = 1;
nn = 1;

% maximal iterations
n_max = 100;

% optimization
while exitflag == 1 && nn <= n_max
    
    alpha_old = alpha;
    
    if L_Phi > L_Phi_max
        alpha_ub = 0.5*(alpha_ub + alpha_lb);
    elseif L_Phi <= L_Phi_max && L_Phi ~= 0
        alpha_lb = 0.5*(alpha_ub + alpha_lb);
    else
        error('error')
    end
    
    alpha = 0.5*(alpha_ub + alpha_lb);
    L_Phi = FcnL_phi(AK,K,P,alpha);
    
    % exit conditions
    if abs(alpha - alpha_old)/abs(alpha_old) <= 10^-12 && L_Phi <= L_Phi_max && L_Phi ~= 0
        exitflag = 0;
    end
    nn = nn + 1;
    
end
clear('alpha_old','alpha_lb','alpha_ub','nn')

alpha

end



function [c, ceq] = nonlinConsAlpha(x, P, alpha)
% All states inside ellipse

    c = x'*P*x - alpha;
    ceq = [];

end

function xdot = system(t, x, u)
% System dynamics

    xdot = zeros(4,1);
    M = .5;
    m = 0.2;
    b = 0.1;
    I = 0.006;
    g = 9.8;
    l = 0.3;

    p = I*(M+m)+M*m*l^2; %denominator for the A and B matrices

    A = [0      1              0           0;
         0 -(I+m*l^2)*b/p  (m^2*g*l^2)/p   0;
         0      0              0           1;
         0 -(m*l*b)/p       m*g*l*(M+m)/p  0];
    B = [     0;
         (I+m*l^2)/p;
              0;
            m*l/p];
    xdot = A*x +B*u;
    
    %mu = 0.5;
    %xdot(1) = x(2) + u(1)*( mu + (1-mu)*x(1) );
    %xdot(2) = x(1) + u(1)*( mu - 4*(1-mu)*x(2) );
    
end

function phi = FcnPhi(x,AK,K)
% Auxiliary function phi(x)

    f = system(0,x,K*x);
    phi = f - AK*x;

end

function L_Phi = FcnL_phi(AK,K,P,alpha)
% Upper bound L_phi

    opt = optimset('MaxFunEvals',10000,'MaxIter',10000,'Display','off');

    [x1,L_Phi_tilde] = fmincon(@(x) -sqrt(FcnPhi(x,AK,K)' * FcnPhi(x,AK,K))/sqrt(x'*x) ,...
        [10;10;10;10],[],[],[],[],[],[],@(x)nonlinConsAlpha(x,P,alpha),opt);
    
    L_Phi = -L_Phi_tilde;

end


