function NN_MPC_regulator

import casadi.*
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
C = [1 0 0 0;
     0 0 1 0];
D = [0;
     0];
% system dimensions
%A = [1 -0.05; 0.05 1];
%B = [0.025; 0.025];
n = length(A(1,:));
m = length(B(1,:)); 

% eqilibilium point
x_eq = zeros(n,1);
u_eq = zeros(m,1);

% Horizon (continuous)
T = 3.0;

% sampling time (Discretization steps)
delta = 0.1;

% Horizon (discrete)
N = T/delta;

% initial conditions
t_0 = 0.0;
x_init = [0.35; 0.35; 0.35;0];

% stage cost
Q = 100* eye(n);
R = 1.0;

% Terminal set and cost
K = [10.0000   15.8365  -79.1980  -18.5048];

P = [1.5979    0.7505   -1.8801   -0.3226;
    0.7505    0.9811   -2.6173   -0.4276;
   -1.8801   -2.6173    9.2988    1.2221;
   -0.3226   -0.4276    1.2221    0.2119];
alpha = 7.236; % obtained via the alternative way

% Set variables for output

% Print Header
fprintf('   k  |      u(k)        x(1)        x(2)     Time \n');
fprintf('---------------------------------------------------\n');

% initilization of measured values
tmeasure = t_0;
xmeasure = x_init;
lb=[repmat([-inf, -inf,-pi/6,-inf], 1, N+1), -200*ones(1,m*N)];
ub=[repmat([inf, inf, pi/6, inf], 1, N+1), 200*ones(1,m*N)];

%get train data
%load train_big_data_small
%draw_trained_data_plot(X_train, U_train, delta)
%[X_train, U_train] = get_train_data(A, B, Q, R, P, K, N, x_eq, u_eq, delta, alpha, n,m,tmeasure, lb, ub, 0.1);
%[X_train_half, U_train_half] = get_train_data(A, B, Q, R, P, K, N, x_eq, u_eq, delta, alpha, n,m,tmeasure, lb, ub, 0.05);
%p = haltonset(4,'Skip',1e3,'Leap',1e2)
%p = scramble(p,'RR2')
%X0 = p(1:2500,:);
%X_train = [];
%for i=1:size(X0,1)
%    X_train = [X_train; [-2 + 4*X0(i,1), -2+4*X0(i,2), -1.04+2.08*X0(i,3),-1.04+2.08*X0(i,4)]];
%end
%U_train = ones(size(X0,1));
%for i=1:size(X0,1)
%    if abs(X_train(i,1)) > 1.8 || abs(X_train(i,2)) > 1.9 || abs(X_train(i,3)) > 1.7 
%        U_train(i) = 250;
%    end
%end
%U_train = get_u_control(A, B, Q, R, P, K, N, x_eq, u_eq, delta, alpha, n,m,tmeasure, lb, ub,X_train)
load train_big_data_small

%[X_train_zero, U_train_zero] = get_train_data_near_zero(A, B, Q, R, P, K, N, x_eq, u_eq, delta, alpha, n,m,tmeasure, lb, ub);
%draw_3d_plot(X_train,U_train);
%uncomment to see the vectors plot with initial state and its next state by
%calcualted U control law
%X_train = [X_train; X_train_zero];
%U_train = [U_train U_train_zero];
%draw_trained_data_plot(X_train, U_train, delta);
%draw_trained_data_plot(X_train_zero, U_train_zero, delta);

mpciterations = 200;

%x_OL = x_init;

%simulate_mpc_simple(A, B, Q, R, P, K, N, x_eq, u_eq, delta, alpha, n,m,tmeasure, lb, ub, x_init, mpciterations);

%while abs(x_OL(1)) > 0.005 || abs(x_OL(2)) > 0.005
%    x_OL = x_init;
%    net = train_nn_regulator(X_train, U_train,n, 25, true,'net_big_data_small.mat');
    
%    x_OL = simulate_mpc(mpciterations, net, x_OL, tmeasure, xmeasure, delta, false);
%end

net = train_nn_regulator(X_train, U_train,n, 25, true,'net_big_data_small.mat');
%simulate_mpc(mpciterations, net, x_init, tmeasure, xmeasure, delta, true);
draw_trajectories(A, B, Q, R, P, K, N, x_eq, u_eq, delta, alpha, n,m,tmeasure, lb, ub, x_init, mpciterations, net);
%sizes_to_check = [5; 8; 10; 15; 20; 30; 40; 50]
%for i=1:size(sizes_to_check)
%    sizes_to_check(i)
%    net = train_nn_regulator(X_train, U_train,n, sizes_to_check(i), false,'');
%    calculate_mean_error(X_train,U_train, net)
%end


%x_OL = simulate_mpc(mpciterations, net, x_init, tmeasure, xmeasure, delta, true);
%svm_cl_upper = get_svm_classifier(X_train(1:2000,:), U_train(1:2000), delta, 2,-0.99, {'Точки c управлением u(t) > -1','Точки c управлением u(t) = -1','Опорные вектора'});
%svm_cl_lower = get_svm_classifier(X_train(1:2000,:), U_train(1:2000), delta, 2,0.99, {'Точки c управлением u(t) < 1','Точки c управлением u(t) = 1','Опорные вектора'});
svm_cl_feas = get_svm_classifier(X_train, U_train, delta, 1,0, {'Точки вне области притяжения','Точки внутри области притяжения','Опорные вектора'});

[predicted_labels, scores] = predict(svm_cl_feas, X_train);

[new_X_train, new_U_train] = filter_feasible_points(X_train, U_train, predicted_labels);
calculate_mean_error(new_X_train,new_U_train,net)
%draw_trained_data_plot(new_X_train, new_U_train, delta)

X = generate_random_points(400, -0.2,0.2,-0.2,0.2);
[predicted_labels_random, scores] = predict(svm_cl_feas, X);
new_X_train_zero = [];
for i= 1:size(X,1)
    if predicted_labels_random(i) == 1
        new_X_train_zero = [new_X_train_zero;X(i,:)];
    end
end
U_law_zero = get_u_control(A, B, Q, R, P, K, N, x_eq, u_eq, delta, alpha, n,m,tmeasure, lb, ub,new_X_train_zero)

net_feas = [];
x_OL = x_init;
while abs(x_OL(1)) > 0.001 || abs(x_OL(2)) > 0.001
    x_OL = x_init;
    net_feas = train_nn_regulator(new_X_train, new_U_train,n,25, false, 'net_feas_model_with_zero.mat');
    
    x_OL = simulate_mpc(mpciterations, net_feas, x_OL, tmeasure, xmeasure, delta, false);
    
end
x_OL = simulate_mpc(mpciterations, net_feas, x_init, tmeasure, xmeasure, delta, true);

end

function [mean_error] = calculate_mean_error(X_train, U_train, net)
    mean_error = 0;
    for i = 1:size(U_train,2)
            u_law = net(X_train(i,:)');
            mean_error = mean_error + (u_law - U_train(i))*(u_law - U_train(i));
    end

    mean_error = mean_error / size(U_train,2);
end

function [X] = generate_random_points(n, x1_l, x1_u,x2_l, x2_u)
    X = [];
    for i = 1:n
        point = rand([1,2]);
        point(1,1) = x1_l + (x1_u-x1_l)*point(1,1);
        point(1,2) = x2_l + (x2_u-x2_l)*point(1,2);
        X = [X; point];
    end
end

function [] = simulate_mpc_simple(A, B, Q, R, P, K, N, x_eq, u_eq, delta, alpha, n,m,tmeasure, lb, ub, x_init, mpciterations)
    t = [];
    x = [];
    u = [];
    %figure;
    x_OL = x_init
    
    for ii = 1:mpciterations % maximal number of iterations
        
        t_Start = tic;
        u_OL=get_control_law(x_OL, A, B, Q, R, P, K, N, x_eq, u_eq, delta, alpha, n,m,tmeasure, lb, ub);
        t_Elapsed = toc( t_Start );
        %%    

        % Store closed loop data
        t = [ t, tmeasure ];
        x = [ x, x_OL ];
        u = [ u, u_OL];

        % Update closed-loop system (apply first control move to system)
        x_OL = dynamic(delta, x_OL, u_OL);
        tmeasure = tmeasure + delta;
        
        % Print numbers
        fprintf(' %3d  | %+11.6f %+11.6f %+11.6f  %+6.3f\n', ii, u(end),...
                     x(1,end), x(2,end),t_Elapsed);

        %plot predicted and closed-loop state trajetories

        plot(x(1,:),x(3,:),'r'), grid on, hold on,
        %plot(x_OL(1:n:n*(N+1)),x_OL(n:n:n*(N+1)),'g') 
        %plot(x(1,:),x(2,:),'ob')
        xlabel('x1(t)')
        ylabel('x3(t)')
        drawnow

    end
    x_OL
    figure;
    stairs(t,u);
    plot(t,x(1,:),'b');
    plot(t,x(2,:),'b');
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

             plot(x(1,:),x(2,:),'b'), grid on, hold on,
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

function [] = draw_trajectories(A, B, Q, R, P, K, N, x_eq, u_eq, delta, alpha, n,m,tmeasure, lb, ub, x_init, mpciterations, net)
    t = [];
    x = [];
    x_net = [];
    u = [];
    u_net = [];
    %figure;
    x_OL = x_init;
    x_OL_net = x_init;
    
    for ii = 1:mpciterations % maximal number of iterations
        
        t_Start = tic;
        u_OL=get_control_law(x_OL, A, B, Q, R, P, K, N, x_eq, u_eq, delta, alpha, n,m,tmeasure, lb, ub);
        u_OL_net = net(x_OL_net);
        t_Elapsed = toc( t_Start );
        %%    

        % Store closed loop data
        t = [ t, tmeasure ];
        x = [ x, x_OL ];
        x_net = [x_net, x_OL_net];
        u = [ u, u_OL];
        u_net = [u_net, u_OL_net];

        % Update closed-loop system (apply first control move to system)
        x_OL = dynamic(delta, x_OL, u_OL);
        x_OL_net = dynamic(delta, x_OL_net, u_OL_net);
        tmeasure = tmeasure + delta;

    end
        figure;
    plot(x(1,:),x(3,:),'r')
    grid on
    hold on
    plot(x_net(1,:),x_net(3,:),'b')
    grid on
    hold on
    xlabel('x(1)')
    ylabel('x(3)')
    legend({'Траектория(стандартный MPC)','Траектория(нейросетевой регулятор)'});
    hold off
    
    figure;
    plot(t,x(1,:),'r')
    grid on
    hold on
    plot(t,x_net(1,:),'b')
    grid on
    hold on
    xlabel('t')
    ylabel('x(1)')
    legend({'Траектория(стандартный MPC)','Траектория(нейросетевой регулятор)'});
    hold off
    
    figure;
    plot(t,x(3,:),'r')
    grid on
    hold on
    plot(t,x_net(3,:),'b')
    grid on
    hold on
    xlabel('t')
    ylabel('x(3)')
    legend({'Траектория(стандартный MPC)','Траектория(нейросетевой регулятор)'});
    hold off
    
    figure;
    plot(t,u,'r')
    grid on
    hold on
    plot(t,u_net,'b')
    grid on
    hold on
    xlabel('t')
    ylabel('u')
    legend({'Управление(стандартный MPC)','Управление(нейросетевой регулятор)'});
    hold off
end

function [new_X_train, new_U_train] = filter_feasible_points(X_train, U_train, predicted_labels)
    new_X_train = [];
    new_U_train = [];
    for i= 1:size(U_train,2)
        if predicted_labels(i) == 1
            new_X_train = [new_X_train;X_train(i,:)];
            new_U_train = [new_U_train U_train(i)];
        end
    end
end

function [svm_cl] = get_svm_classifier(X_train, U_train, delta, mode, u_eq, legend_text)
    data1 = [];
    data2 = [];
    labels = [];
    
    use_saved_data = false;
    if use_saved_data
       load svm_labels
    else
       labels = get_labels_for_states(X_train, U_train, delta);
    end
    
    if mode == 2
        new_labels = [];
        use_saved_data = false;
        if use_saved_data
            load svm_labels
        else
            for i= 1:size(U_train,2)
                res = -1;
                if (u_eq < 0 && U_train(i) <= u_eq + 0.0001 || u_eq > 0 && U_train(i) >= u_eq -0.0001) && labels(i) == 1
                    res = 1;
                end
                new_labels = [new_labels; res]; 
            end
            labels = new_labels;
        end
    end
    
    for i=1:size(U_train,2)
        if labels(i) == -1
            data1 = [data1;X_train(i,:)];
        else
            data2 = [data2;X_train(i,:)];
        end
    end

    figure;
    
    plot3(data1(:,1),data1(:,2),data1(:,3),'r.','MarkerSize',15)
    hold on
    plot3(data2(:,1),data2(:,2),data2(:,3),'b.','MarkerSize',15)
    
    axis equal
    legend({'Точки вне области притяжения','Точки из области притяжения'});
    hold off

    %Train the SVM Classifier
    svm_cl = fitcsvm(X_train,labels,'KernelFunction','rbf',...
         'BoxConstraint',10000,'ClassNames',[-1,1]);

    % Predict scores over the grid
    d = 0.05;
    [x1Grid,x2Grid, x3Grid] = meshgrid(min(X_train(:,1)):d:max(X_train(:,1)),...
        min(X_train(:,2)):d:max(X_train(:,2)), min(X_train(:,3)):d:max(X_train(:,3)));
    xGrid = [x1Grid(:),x2Grid(:),x3Grid(:)];
    xGrid = [xGrid,zeros(size(xGrid,1),1)];
    [predicted_labels,scores] = predict(svm_cl,xGrid);

    % Plot the data and the decision boundary
    figure;
    h(1:2) = gscatter(X_train(:,1),X_train(:,2),labels,'rb','.');
    hold on
    h(3) = plot(X_train(svm_cl.IsSupportVector,1),X_train(svm_cl.IsSupportVector,2),'ko');
    contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0],'k');
    legend(h,legend_text);
    axis equal
    hold off
end

function [] = draw_3d_plot(X_train,U_train)
    [x1_arr,x2_arr] = meshgrid(-2:0.05:0.5,-1:0.05:2);
    
    x3_arr = reshape(U_train, size(x1_arr,1), size(x1_arr,2));
    figure;
    surf(x1_arr, x2_arr, x3_arr);
    xlabel('x(1)')
    ylabel('x(2)')
    zlabel('u')
end

function [U_train] = get_u_control(A, B, Q, R, P, K, N, x_eq, u_eq, delta, alpha, n,m,tmeasure, lb, ub, X)
    U_train = [];
    use_saved_data = false;
    if use_saved_data
        %
    else
        for i=1:size(X,1)
            U_train = [U_train get_control_law(X(i,:), A, B, Q, R, P, K, N, x_eq, u_eq, delta, alpha, n,m,tmeasure, lb, ub)];
        end
    end
end

function [X_train, U_train] = get_train_data(A, B, Q, R, P, K, N, x_eq, u_eq, delta, alpha, n,m,tmeasure, lb, ub, step)
    X_train = [];
    U_train = [];
    use_saved_data = false;
    if use_saved_data
        load new_train_data_0_05 %train_data_0.1.mat
        if exist('X_train_half','var')
            X_train = eval('X_train_half');
        end
        if exist('U_train_half','var')
            U_train = eval('U_train_half');
        end
    else
        for x1 = -0.5:step:0.5
            for x2 = -0.5:step:0.5
                for x3 = -0.5:step:0.5
                    for x4 = -0.5:step:0.5
               X_train = [X_train; x1 x2 x3 x4;];
               U_train = [U_train get_control_law([x1;x2;x3;x4], A, B, Q, R, P, K, N, x_eq, u_eq, delta, alpha, n,m,tmeasure, lb, ub)];
                    end
                end
            end
        end
    end
end

function [X_train, U_train] = get_train_data_near_zero(A, B, Q, R, P, K, N, x_eq, u_eq, delta, alpha, n,m,tmeasure, lb, ub)
    X_train = [];
    U_train = [];
    use_saved_data = false;
    if use_saved_data
        load train_zero.mat
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

function net = train_nn_regulator(X_train, U_train, n, neurons_count, use_saved_data, filename)
     if use_saved_data
        load(filename) %net_example_6
        if exist('net_feas','var')
            net = eval('net_feas')
        end
     else
          X_train = X_train';
          x = {X_train(1,:); X_train(2,:);X_train(3,:);X_train(4,:)};
          net = feedforwardnet;
          net.numinputs = n;
          net.trainFcn = 'trainlm';
          net.layers{1}.transferFcn = 'logsig';
          net.layers{1}.size = neurons_count;
          net.inputConnect = [1 1 1 1; 0 0 0 0];
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
         
         plot([X_train_set_init(i, 1), X_train_set_next(i, 1)],[X_train_set_init(i, 3), X_train_set_next(i, 3)],'b'), grid on, hold on,
         xlabel('x(1)')
         ylabel('x(3)')
         drawnow
    end
end

function [labels] = get_labels_for_states(X_train, U_train, delta)
    labels = [];
    for i= 1:size(U_train,2)
        new_x = dynamic(delta, X_train(i, :)', U_train(i));
        res = 1;
        %if new_x(3) < -pi/6 || new_x(3) > pi/6 || U_train(i) < -200 || U_train(i) > 2000
        %    res = -1;
        %end
        %if abs(new_x(1)) > 0.5  || abs(new_x(2)) > 0.5 || abs(new_x(3)) > 0.5
        %    res = -1;
        %end
        if abs(U_train(i)) >200
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
    import casadi.*
    
    con_bound=zeros(N*n,1);
    con_lb=[con_bound;-inf];
    con_ub=[con_bound;alpha];
    y=MX.sym('y',N*m+(N+1)*n);
    obj=costfunction(N, y, x_eq, u_eq, Q_cost, R_cost, P,n,m,delta);
    con=nonlinearconstraints(N, delta, y, x_eq, P, alpha,n,m);
    nlp = struct('x', y, 'f', obj, 'g', con);
    solver = nlpsol('solver', 'ipopt', nlp); %,'file_print_level',5
    Aeq=[eye(n),zeros(n,N*(n+m))];
    beq=x_init;
    lb(1:n)=x_init;
    ub(1:n)=x_init;
    
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
    %[y_OL, V, exitflag, output]=fmincon(@(y) costfunction( N, y, x_eq, u_eq, Q_cost, R_cost, P,n,m,delta),...
    %    y_init,[],[],Aeq,beq,lb,ub,...
    %    @(y) nonlinearconstraints(N, delta, y, x_eq, P, alpha,n,m), options);
    res = solver('x0' , y_init,... % solution guess
             'lbx', lb,...           % lower bound on x
             'ubx', ub,...           % upper bound on x
             'lbg', con_lb,...           % lower bound on g
             'ubg', con_ub);             % upper bound on g
    y_OL=full(res.x); 
    x_OL=y_OL(1:n*(N+1));
    u_OL=y_OL(n*(N+1)+1:end);
    
    t_Elapsed = toc( t_Start ); 
    u_control = u_OL(1:m);
end

function xdot = system(~, x, u)
    % Systemn dynamics
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
    
    xdot = zeros(4,1);
    xdot = A*x +B*u;
   % mu = 0.5;
    %xdot(1) = -x(2) + 0.5*(1+x(1))*u(1);
    %xdot(2) = x(1) + 0.5*(1-4*x(2))*u(1);
    
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


  function [ceq] = nonlinearconstraints(N, delta, y, x_eq, P, alpha,n,m) 
   % Introduce the nonlinear constraints also for the terminal state
   
   x=y(1:n*(N+1));
   u=y(n*(N+1)+1:end);
   ceq = [];
   % constraints along prediction horizon
    for k=1:N
        x_k=x((k-1)*n+1:k*n);
        x_new=x(k*n+1:(k+1)*n);        
        u_k=u((k-1)*m+1:k*m);
        %dynamic constraint
        ceqnew=x_new - dynamic(delta, x_k, u_k);
        ceq = [ceq; ceqnew];
        %nonlinear constraints on state and input could be included here
    end
   %
   %terminal constraint
   [cnew] = terminalconstraints( x(n*N+1:n*(N+1)), x_eq, P, alpha);
    ceq = [ceq; cnew];
    
end

function cost = terminalcosts(x, x_eq, P)
    % Introduce the terminal cost
    cost = (x-x_eq)'*P*(x-x_eq);
end


function [c] = terminalconstraints(x, x_eq, P, alpha)
    % Introduce the terminal constraint
    c   = (x-x_eq)'*P*(x-x_eq) - alpha;
end


function [x] = dynamic(delta, x0, u)
    x=x0+delta*system(0,x0,u);
end