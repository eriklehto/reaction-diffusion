function [L,R] = rbf_hfd_laplace_beltrami(x,ep,n,m,p,nrml,imax,jmax)
%% [L,R] = rbf_hfd_laplace_beltrami(x,ep,n,m,p,nrml,imax,jmax)
% Create differentiation matrices for the Laplace-Beltrami operator on a
% surface defined by nodes x and unit normals nrml (at x), using a simple
% greedy algorithm for selecting stencils with stable weights.
%
% Inputs:
% n - number of explicit nodes in stencil (including center)
% m - number of implicit nodes in stencil (EXCLUDING center)
% ep - shape parameter (scalar)
% p - neighborhood size
% 
% Optional parameters for greedy algorithm:
% imax - number of candidates for each explicit stencil (default: 30)
% jmax - number of candidates for each implicit stencil (default: 30)
%
% Written by Erik Lehto, 2017
%

% Set default values
if nargin<7
    imax = 30; jmax = 30;
elseif nargin<8
    jmax = 30;
end

% Local index sets
np = n+3; mp = m+3;
P = [ones(nchoosek(np-1,n-1),1), nchoosek(2:np,n-1)];
Q = nchoosek(2:mp+1,m);

imax = min(imax,size(P,1));
jmax = max(1,min(jmax,size(Q,1)));
disp(['# of explicit/implicit candidate stencils: ' num2str([imax jmax]) ])

% Global index sets
N = size(x,1);
I = repmat(1:N,p,1)';
[J,dist] = knnsearch(x,x,'k',p);

% Space for weights
WL = zeros(N,p);
WR = zeros(N,p);
condA = zeros(N,1);

% Counters
rej = 0; nn = 0; iter = 0;
for k = 1:N
    xk = x(J(k,:),:);
    nr = nrml(J(k,:),:);
    
    xij = repmat(xk(:,1),[1 p]); xij = xij-xij.'; nxi = repmat(nr(:,1),[1 p]);
    yij = repmat(xk(:,2),[1 p]); yij = yij-yij.'; nyi = repmat(nr(:,2),[1 p]);
    zij = repmat(xk(:,3),[1 p]); zij = zij-zij.'; nzi = repmat(nr(:,3),[1 p]);
    
    r2 = xij.^2 + yij.^2 + zij.^2;
    
    %%% Compute all necessary ingredients for assembling the Hermite system
    A = exp(-ep^2*r2);
    dr = -2*ep^2*exp(-ep^2*r2);
    condA(k) = cond(A);
    
    Bx = ((1-nxi.^2).*xij - nxi.*nyi.*yij - nxi.*nzi.*zij).*dr;
    By = (-nxi.*nyi.*xij + (1-nyi.^2).*yij - nyi.*nzi.*zij).*dr;
    Bz = (-nxi.*nzi.*xij - nyi.*nzi.*yij + (1-nzi.^2).*zij).*dr;

    L = chol(A,'lower');
    
    Dx = (Bx/L.')/L;
    Dy = (By/L.')/L;
    Dz = (Bz/L.')/L;
    
    B1 = (Dx*Bx + Dy*By + Dz*Bz);
    B2 = B1.';
    C = B1/L.'; C = C*C.';
            
    %%% Sort candidate stencils by mean distance
    dP = reshape(dist(k,P),size(P));
    [~,s] = sort(mean(dP,2)); % sort stencils by mean distance to center  
    Pk = P(s,:);
    dQ = reshape(dist(k,Q),size(Q));
    [~,s] = sort(mean(dQ,2));
    Qk = Q(s,:);

    flag = 0;
    i = 1; j = 1;
    warning('off','MATLAB:nearlySingularMatrix');
    while ~flag && i<=imax && j<=jmax
        iter = iter+1;
        
        in = Pk(i,:);
        im = Qk(j,:);
        
        % Hermite interpolation system and right hand side
        Ah = [A(in,in) B2(in,im) ones(n,1); ...
            B1(im,in), C(im,im), zeros(m,1); ...
            ones(1,n), zeros(1,m), 0];
        b = [B1(1,in), C(1,im), 0]';
        
        w = Ah\b;
        
        wl = zeros(p,1);
        wl(in) = w(1:n);
        wr = eye(p,1); wr(im) = -w(n+1:n+m);
        wl = wl/sum(wr); wr = wr/sum(wr);
        
        ddl = all(wl(2:end)>-1e2*eps);       % diagonal dominance of wl
        ddr = wr(1)>sum(abs(wr(2:end)));     % diagonal dominance of wr
        posr = all(wr>-1e2*eps);             % positivity of wr
        
        % Set flag=1 if weights are stable
        if m<10 
            flag = ddl && ddr && posr;
        else
            flag = ddl && ddr;
        end
        
        if (i==1 && j==1) || flag
            % Save weights
            WL(k,:) = wl;
            WR(k,:) = wr;
        end
        if ~flag
            j = j+1;
            if j>jmax
                i = i+1; j = 1;
            end
        end
    end
    if ~flag, rej = rej+1;
    elseif (i==1 && j==1), nn = nn+1;
    end
end
warning('on','MATLAB:nearlySingularMatrix');
disp(['[min mean max] cond(A): ' num2str([min(condA) mean(condA) max(condA)],' %.1e') ]);
disp(['Rejected stencils: ' num2str(rej) ', evaluated stencils: ' num2str(iter)]);
disp(['Nearest neighbor stencils: ' num2str(nn) '/' num2str(N)]);

L = sparse(I,J,WL,N,N);
R = sparse(I,J,WR,N,N);