function NN_MPC_regulator

% clear workspace, close open figures
clear all
close all
clc

% Set options 
tol_opt       = 1e-3;
options = optimset('Display','off',...
                'TolFun', tol_opt,...
                'MaxIter', 10000,...
                'Algorithm', 'active-set',...
                'FinDiffType', 'forward',...
                'RelLineSrchBnd', [],...
                'RelLineSrchBndDuration', 1,...
                'TolConSQP', 1e-6);

% system dimensions
A = [1 -0.05; 0.05 1];
B = [0.025; 0.025];
n = length(A(1,:));
m = length(B(1,:)); 

% eqilibilium point
x_eq = zeros(n,1);
u_eq = zeros(m,1);

% Horizon (continuous)
T = 1.0;

% sampling time (Discretization steps)
delta = 0.1;

% Horizon (discrete)
N = T/delta;

% initial conditions
t_0 = 0.0;
x_init = [-0.7; -0.8];

% stage cost
Q = 0.1*eye(n);
R = 1.0;

% Terminal set and cost
K = [-0.3799 -0.4201];
P = [48.3043, -2.53;
    -2.53, 70.8191];

alpha = 0.2129; % obtained via the alternative way

% Set variables for output

% Print Header
fprintf('   k  |      u(k)        x(1)        x(2)     Time \n');
fprintf('---------------------------------------------------\n');

% initilization of measured values
tmeasure = t_0;
xmeasure = x_init;
lb=[repmat([-2 -1], 1, N+1), -inf*ones(1,m*N)];
ub=[repmat([0.5 2], 1, N+1), inf*ones(1,m*N)];

%get train data
[X_train, U_train] = get_train_data(A, B, Q, R, P, K, N, x_eq, u_eq, delta, alpha, n,m,tmeasure, lb, ub, 0.1);
[X_train_half, U_train_half] = get_train_data(A, B, Q, R, P, K, N, x_eq, u_eq, delta, alpha, n,m,tmeasure, lb, ub, 0.05);
[X_train_zero, U_train_zero] = get_train_data_near_zero(A, B, Q, R, P, K, N, x_eq, u_eq, delta, alpha, n,m,tmeasure, lb, ub);
%uncomment to see the vectors plot with initial state and its next state by
%calcualted U control law
%X_train = [X_train; X_train_zero];
%U_train = [U_train U_train_zero];
draw_trained_data_plot(X_train, U_train, delta);
%draw_trained_data_plot(X_train_zero, U_train_zero, delta);

% create network 
net = train_nn_regulator(X_train, U_train,n, true, 'net_example_6.mat');

mpciterations = 200;
% Set variables for output

x_OL = x_init;

while abs(x_OL(1)) > 0.001 || abs(x_OL(2)) > 0.001
    x_OL = x_init;
    net = train_nn_regulator(X_train, U_train,n, false,'net_example_6.mat');
    
    x_OL = simulate_mpc(mpciterations, net, x_OL, tmeasure, xmeasure, delta, false);
    
end

x_OL = simulate_mpc(mpciterations, net, x_init, tmeasure, xmeasure, delta, true);

svm_cl = get_svm_classifier(X_train, U_train, delta);

[predicted_labels, scores] = predict(svm_cl, X_train);

[new_X_train, new_U_train] = filter_feasible_points(X_train, U_train, predicted_labels);
draw_trained_data_plot(new_X_train, new_U_train, delta)
net_feas = [];
x_OL = x_init;
while abs(x_OL(1)) > 0.001 || abs(x_OL(2)) > 0.001
    x_OL = x_init;
    net_feas = train_nn_regulator(new_X_train, new_U_train,n, false, 'net_feas_model_with_zero.mat');
    
    x_OL = simulate_mpc(mpciterations, net_feas, x_OL, tmeasure, xmeasure, delta, false);
    
end
x_OL = simulate_mpc(mpciterations, net_feas, x_init, tmeasure, xmeasure, delta, true);
end

function [x_OL] = simulate_mpc(mpciterations, net, x_OL, tmeasure, ~, delta, draw_plot)
    t = [];
    x = [];
    u = [];
    if draw_plot
        figure;
    end
    for ii = 1:mpciterations % maximal number of iterations
        
        t_Start = tic;

        u_OL=net(x_OL);
        t_Elapsed = toc( t_Start );
        %%    

        % Store closed loop data
        t = [ t, tmeasure ];
        x = [ x, x_OL ];
        u = [ u, u_OL];

        % Update closed-loop system (apply first control move to system)
        x_OL = dynamic(delta, x_OL, u_OL);
        tmeasure = tmeasure + delta;
        
        if draw_plot
            % Print numbers
             fprintf(' %3d  | %+11.6f %+11.6f %+11.6f  %+6.3f\n', ii, u(end),...
                     x(1,end), x(2,end),t_Elapsed);

             %plot predicted and closed-loop state trajetories

             plot(x(1,:),x(2,:),'r'), grid on, hold on,
             %plot(x_OL(1:n:n*(N+1)),x_OL(n:n:n*(N+1)),'g') plot(x(1,:),x(2,:),'ob')
             xlabel('x(1)')
             ylabel('x(2)')
             drawnow
        end

    end
    x_OL
    if draw_plot
        figure;
        stairs(t,u);
    end
end

function [new_X_train, new_U_train] = filter_feasible_points(X_train, U_train, predicted_labels)
    new_X_train = [];
    new_U_train = [];
    size(U_train)
    for i= 1:size(U_train,2)
        if predicted_labels(i) == 1
            new_X_train = [new_X_train;X_train(i,:)];
            new_U_train = [new_U_train U_train(i)];
        end
    end
end

function [svm_cl] = get_svm_classifier(X_train, U_train, delta)
    use_saved_data = false;
    if use_saved_data
        load svm_labels
    else
        labels = get_labels_for_states(X_train, U_train, delta);
    end
    
    data1 = [];
    data2 = [];
    for i=1:size(U_train,2)
        if labels(i) == -1
            data1 = [data1;X_train(i,:)];
        else
            data2 = [data2;X_train(i,:)];
        end
    end

    figure;
    plot(data1(:,1),data1(:,2),'r.','MarkerSize',15)
    hold on
    plot(data2(:,1),data2(:,2),'b.','MarkerSize',15)
    axis equal
    hold off

    %Train the SVM Classifier
    svm_cl = fitcsvm(X_train,labels,'KernelFunction','rbf',...
        'BoxConstraint',Inf,'ClassNames',[-1,1]);

    % Predict scores over the grid
    d = 0.02;
    [x1Grid,x2Grid] = meshgrid(min(X_train(:,1)):d:max(X_train(:,1)),...
        min(X_train(:,2)):d:max(X_train(:,2)));
    xGrid = [x1Grid(:),x2Grid(:)];
    [~,scores] = predict(svm_cl,xGrid);

    % Plot the data and the decision boundary
    figure;
    h(1:2) = gscatter(X_train(:,1),X_train(:,2),labels,'rb','.');
    hold on
    h(3) = plot(X_train(svm_cl.IsSupportVector,1),X_train(svm_cl.IsSupportVector,2),'ko');
    contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0],'k');
    legend(h,{'-1','+1','Support Vectors'});
    axis equal
    hold off
end

function [X_train, U_train] = get_train_data(A, B, Q, R, P, K, N, x_eq, u_eq, delta, alpha, n,m,tmeasure, lb, ub, step)
    X_train = [];
    U_train = [];
    use_saved_data = false;
    if use_saved_data
        load train_data_0.1.mat
    else
        for x1 = -2:step:0.5
            for x2 = -1:step:2
               X_train = [X_train; x1 x2;];
               U_train = [U_train get_control_law([x1;x2], A, B, Q, R, P, K, N, x_eq, u_eq, delta, alpha, n,m,tmeasure, lb, ub)];
            end
        end
    end
end

function [X_train, U_train] = get_train_data_near_zero(A, B, Q, R, P, K, N, x_eq, u_eq, delta, alpha, n,m,tmeasure, lb, ub)
    X_train = [];
    U_train = [];
    use_saved_data = true;
    if use_saved_data
        load train_data_near_zero.mat
        if exist('X_train_zero','var')
            X_train = eval('X_train_zero');
        end
        if exist('U_train_zero','var')
            U_train = eval('U_train_zero');
        end
    else
        for x1 = -0.1:0.01:0.1
            for x2 = -0.1:0.01:0.1
               X_train = [X_train; x1 x2;];
               U_train = [U_train get_control_law([x1;x2], A, B, Q, R, P, K, N, x_eq, u_eq, delta, alpha, n,m,tmeasure, lb, ub)];
            end
        end
    end
end

function net = train_nn_regulator(X_train, U_train, n, use_saved_data, filename)
     if use_saved_data
        load(filename) %net_example_6
        if exist('net_feas','var')
            net = eval('net_feas')
        end
     else
          X_train = X_train';
          x = {X_train(1,:); X_train(2,:)};
          net = feedforwardnet;
          net.numinputs = n;
          net.trainFcn = 'trainlm';
          net.layers{1}.transferFcn = 'logsig';
          net.layers{1}.size = 22;
          net.inputConnect = [1 1 ; 0 0 ];
          net = configure(net,x);
          net = train(net,x,U_train)
     end   
end

function res = draw_trained_data_plot(X_train, U_train, delta)
    X_train_set_init = X_train;
    X_train_set_next = [];
    figure;
    for i= 1:size(U_train,2)
         X_train_set_next = [X_train_set_next; dynamic(delta, X_train_set_init(i, :)', U_train(i))']; 
         
         plot([X_train_set_init(i, 1), X_train_set_next(i, 1)],[X_train_set_init(i, 2), X_train_set_next(i, 2)],'b'), grid on, hold on,
         xlabel('x(1)')
         ylabel('x(2)')
         drawnow
    end
end

function [labels] = get_labels_for_states(X_train, U_train, delta)
    labels = [];
    for i= 1:size(U_train,2)
        new_x = dynamic(delta, X_train(i, :)', U_train(i));
        res = 1;
        if new_x(1) < -2 || new_x(1) > 0.5 || new_x(2) < -1 || new_x(2) > 2
            res = -1;
        end
        labels = [labels; res]; 
    end
end

function u_control = get_control_law(x_init, Ac, Bc, Q_cost,R_cost, P,K, N, x_eq, u_eq, delta, alpha, n,m,tmeasure, lb, ub)
    tol_opt       = 1e-3;
    options = optimset('Display','off',...
                'TolFun', tol_opt,...
                'MaxIter', 15000,...
                'Algorithm', 'active-set',...
                'FinDiffType', 'forward',...
                'RelLineSrchBnd', [],...
                'RelLineSrchBndDuration', 1,...
                'TolConSQP', 1e-6);
    Aeq=[eye(n),zeros(n,N*(n+m))];
    beq=x_init;
    
    u0 = 0.5*ones(m*N,1);
    x0=zeros(n*(N+1),1);
    x0(1:n) = x_init;
    for k=1:N
        x0(n*k+1:n*(k+1)) = dynamic(delta,x0(n*(k-1)+1:n*k), u0(k));
    end
    
    y_init=[x0;u0];
    
    t_Start = tic;
    
    % Solve optimization problem
    %structure: y_OL=[x_OL,u_OL];
    [y_OL, V, exitflag, output]=fmincon(@(y) costfunction( N, y, x_eq, u_eq, Q_cost, R_cost, 7.4014*P,n,m,delta),...
        y_init,[],[],Aeq,beq,lb,ub,...
        @(y) nonlinearconstraints(N, delta, y, x_eq, 7.4014*P, alpha,n,m), options);
    x_OL=y_OL(1:n*(N+1));
    u_OL=y_OL(n*(N+1)+1:end);
    t_Elapsed = toc( t_Start ); 
    u_control = u_OL(1:m);
end

function xdot = system(~, x, u)
    % Systemn dynamics
    xdot = zeros(2,1);

    mu = 0.5;
    xdot(1) = -x(2) + 0.5*(1+x(1))*u(1);
    xdot(2) = x(1) + 0.5*(1-4*x(2))*u(1);
    
end

function cost = costfunction(N, y, x_eq, u_eq, Q, R, P,n,m,delta)
    % Formulate the cost function to be minimized
    
    cost = 0;
    x=y(1:n*(N+1));
    u=y(n*(N+1)+1:end);
    
    % Build the cost by summing up the stage cost and the
    % terminal cost
    for k=1:N
        x_k=x(n*(k-1)+1:n*k);
        u_k=u(m*(k-1)+1:m*k);
        cost = cost + delta*runningcosts(x_k, u_k, x_eq, u_eq, Q, R);
    end
    cost = cost + terminalcosts( x(n*N+1:n*(N+1)), x_eq, P);
    
end

function cost = runningcosts(x, u, x_eq, u_eq, Q, R)
    % Provide the running cost   
    cost = (x-x_eq)'*Q*(x-x_eq) + (u-u_eq)'*R*(u-u_eq);
    
end


  function [c, ceq] = nonlinearconstraints(N, delta, y, x_eq, P, alpha,n,m) 
   % Introduce the nonlinear constraints also for the terminal state
   
   x=y(1:n*(N+1));
   u=y(n*(N+1)+1:end);
   c = [];
   ceq = [];
   % constraints along prediction horizon
    for k=1:N
        x_k=x((k-1)*n+1:k*n);
        x_new=x(k*n+1:(k+1)*n);        
        u_k=u((k-1)*m+1:k*m);
        %dynamic constraint
        ceqnew=x_new - dynamic(delta, x_k, u_k);
        ceq = [ceq ceqnew];
        %nonlinear constraints on state and input could be included here
    end
   %
   %terminal constraint
   [cnew, ceqnew] = terminalconstraints( x(n*N+1:n*(N+1)), x_eq, P, alpha);
    c = [c cnew];
    ceq = [ceq ceqnew];
    
end

function cost = terminalcosts(x, x_eq, P)
    % Introduce the terminal cost
    cost = (x-x_eq)'*P*(x-x_eq);
end


function [c, ceq] = terminalconstraints(x, x_eq, P, alpha)
    % Introduce the terminal constraint
    c   = (x-x_eq)'*P*(x-x_eq) - alpha;
    ceq = [];
end


function [x] = dynamic(delta, x0, u)
    x=x0+delta*system(0,x0,u);
end