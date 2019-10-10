clc;clear
addpath('./ClusteringMeasure');
addpath('./code_coregspectral');
addpath('./LRR');
addpath('./twist');
rand('seed',100);

projev = 1.5;

%% COIL-20
dataset='COIL20_3VIEWS.mat';
numClust=20;
num_views=3;
load(dataset);
gt=Y;
clear Y

data{1}=X1;
data{2}=X2;
data{3}=X3;

X1=data{1};
X2=data{2};
X3=data{3};

ratio=1;
sigma(1)=ratio*optSigma(X1);
sigma(2)=ratio*optSigma(X2);
sigma(3)=ratio*optSigma(X3);

cls_num = length(unique(gt));
tic
%% Construct kernel and transition matrix
K=[];
T=cell(1,num_views);
for j=1:num_views
    options.KernelType = 'Gaussian';
    options.t = sigma(j);
    K(:,:,j) = constructKernel(data{j},data{j},options);
    D=diag(sum(K(:,:,j),2));
    L_rw=D^-1*K(:,:,j);
    T{j}=L_rw;
end
T_tensor = cat(3, T{:,:});
t = T_tensor(:);

%% init evaluation result
best_single_view.nmi=0;
feature_concat.nmi=0;
kernel_addition.nmi=0;
markov_mixture.nmi=0;
co_reg.nmi=0;
markov_ag.nmi=0;

V = length(data); 
N = size(data{1},1); % number of samples

for k=1:V
    Z{k} = zeros(N,N); 
    Y{k} = zeros(N,N);
    E{k} = zeros(N,N); 
end
Z_tensor = cat(3, Z{:,:});
E_tensor = cat(3, E{:,:});

y = zeros(N*N*V,1);
dim1 = N;dim2 = N;dim3 = V;
myNorm = 'tSVD_1';
sX = [N, N, V];

tol = 1e-6;
lambda = 0.004;

iter = 0;
mu = 10e-3; 
max_mu = 10e10; 
pho_mu = 2;
max_iter=200;

while iter < max_iter
    Zpre=Z_tensor;
    Epre=E_tensor;
    fprintf('----processing iter %d--------\n', iter+1);
    %% update Z
    Y_tensor = cat(3, Y{:,:});
    y = Y_tensor(:);
    e = E_tensor(:);
    
    [z, objV] = wshrinkObj(t - e + 1/mu*y,1/mu,sX,0,3)   ;
    Z_tensor = reshape(z, sX);
    Z{1}=Z_tensor(:,:,1);
    Z{2}=Z_tensor(:,:,2);
    Z{3}=Z_tensor(:,:,3);
    %% update E
    F = [T{1}-Z{1}+Y{1}/mu;T{2}-Z{2}+Y{2}/mu;T{3}-Z{3}+Y{3}/mu];
    [Econcat] = solve_l1l2(F,lambda/mu);

    E{1} = Econcat(1:size(T{1},1),:);
    E{2} = Econcat(size(T{1},1)+1:size(T{1},1)+size(T{2},1),:);
    E{3} = Econcat(size(T{1},1)+size(T{2},1)+1:end,:);
    E_tensor = cat(3, E{:,:});
    
    for k=1:V
        Y{k} = Y{k} + mu*(T{k}-Z{k}-E{k});
    end
    
    %% check convergence
    leq = T_tensor-Z_tensor-E_tensor;
    leqm = max(abs(leq(:)));
    difZ = max(abs(Z_tensor(:)-Zpre(:)));
    difE = max(abs(E_tensor(:)-Epre(:)));
    err = max([leqm,difZ,difE]);
    fprintf('iter = %d, mu = %.3f, difZ = %.3f, difE = %.8f,err=%d\n'...
            , iter,mu,difZ,difE,err);
    if err < tol
        break;
    end

    iter = iter + 1;
    mu = min(mu*pho_mu, max_mu);
end
toc
S = zeros(N,N);
for k=1:num_views
    S = S + Z{k};
end
[pi,~]=eigs(S',1);
Dist=pi/sum(pi);
pi=diag(Dist);
P_hat=(pi^0.5*S*pi^-0.5+pi^-0.5*S'*pi^0.5)/2;
% toc
[V,Eval,F,P,R,nmi,avgent,AR,ACC,C] = baseline_spectral_onRW_acc(P_hat,numClust,gt,projev);
fprintf('lambda=%f, F=%f, P=%f, R=%f, nmi score=%f, avgent=%f,  AR=%f, ACC=%f,\n',lambda,F(1),P(1),R(1),nmi(1),avgent(1),AR(1),ACC(1));
