% ------------------------------------------------------
%   Reaction-diffusion on surfaces
% ------------------------------------------------------
%
% Written by Erik Lehto, 2017
%

disp('-- Setting RBF-HFD parameters --------------------');
% Fourth order scheme
n = 10; m = 6; p = 19; ep = 3; imax = 15; jmax = 15; 
% Sixth order scheme
% n = 12; m = 15; p = 32; ep = 5; imax = 30; jmax = 30;

dt = 0.1;
tol = 1e-8;             % tolerance for Krylov solver
pattern = 'spots';    % spots or stripes

disp('-- Loading nodes and normals ---------------------');
% load nodes (x), normals (nrml) and triangulation (for visualization)
load('rbc4960.mat');
% load('tooth11930.mat'); 
N = size(x,1);

disp('-- Computing differentiation matrices ------------');
[L,R] = rbf_hfd_laplace_beltrami(x,ep,n,m,p,nrml,imax,jmax);

disp('-- Setting problem parameters --------------------');
d = 0.516;
% Turing pattern initial solution
u0 = rand(N,1)-0.5; v0 = rand(N,1)-0.5;
if strcmp(pattern,'spots')
    del = 0.0045; tau1 = 0.02; tau2 = 0.2; % spots
    cmin = -6; cmax = 15;
    u0 = 5*u0; v0 = 5*v0;
    T = 200;
else
    del = 0.0021; tau1 = 3.5; tau2 = 0; % stripes
    cmin = -0.25; cmax = 0.25;
    u0 = 0.5*u0; v0 = 0.5*v0;
    T = 4000;
end
delu = del*d; delv = del;
alp = 0.899; bet = -0.91;
gam = -alp;
fu = @(t,u,v) alp*u.*(1-tau1*v.^2)+v.*(1-tau2*u);
fv = @(t,u,v) bet*v.*(1+alp*tau1/bet*u.*v)+u.*(gam+tau2*v);

disp('-- Plotting initial solution ---------------------');
figure;
trisurf(tri,x(:,1),x(:,2),x(:,3),u0);
caxis([cmin cmax]);
shading interp; view([-32 22]);
axis equal; axis off;
lighting gouraud; camlight left;
title(['t = ' num2str(0) ', T = ' num2str(T)]);
pause(0.01);

s = 3; % order of SBDF-method
u = zeros(N,s); v = zeros(N,s);
t = (0:s-1)*dt;
u(:,1) = u0; v(:,1) = v0;

disp('-- Initializing SBDF-3 using ode45 ---------------');
U0 = [u0;v0];
rhs = @(t,U) [delu*(L*U(1:N,:))+fu(t,U(1:N,:),U(N+1:2*N,:));...
        delv*(L*U(N+1:2*N,:))+fv(t,U(1:N,:),U(N+1:2*N,:))];

[~,U] = ode45(rhs,(0:s-1)*dt,U0);
u(:,1:s) = U(1:s,1:N)'; v(:,1:s) = U(1:s,N+1:2*N)';

disp('-- Setting up iterative solver -------------------');
maxit = 1e3;
setup.type = 'nofill'; setup.milu = 'row';

% (S)BDF-3 iteration matrices
Lbaru = 11/6*R-delu*dt*L;
Lbarv = 11/6*R-delv*dt*L;

[LDu,UDu] = ilu(Lbaru,setup);
[LDv,UDv] = ilu(Lbarv,setup);

disp('-- Beginning time stepping -----------------------');
msg_length = 0;
fprintf(1,'Time step: ');
for k = s:round(T/dt)
    % Compute right-hand side
    rhsu = 3*u(:,3)-3/2*u(:,2)+1/3*u(:,1)+...
        dt*(3*fu(t(3),u(:,3),v(:,3))-3*fu(t(2),u(:,2),v(:,2))+fu(t(1),u(:,1),v(:,1)));
    rhsv = 3*v(:,3)-3/2*v(:,2)+1/3*v(:,1)+...
        dt*(3*fv(t(3),u(:,3),v(:,3))-3*fv(t(2),u(:,2),v(:,2))+fv(t(1),u(:,1),v(:,1)));

    u(:,1:s-1) = u(:,2:s); % shift
    v(:,1:s-1) = v(:,2:s);
    t(1:s-1) = t(2:s);
    
    % Compute new solution using BiCGSTAB
    [u(:,s),flag] = bicgstab(Lbaru,R*rhsu,tol,maxit,LDu,UDu,u(:,s-1));
    [v(:,s),~] = bicgstab(Lbarv,R*rhsv,tol,maxit,LDv,UDv,v(:,s-1));
    t(s) = t(s-1) + dt;

    if mod(k,20) == 0
        % Plot solution
        trisurf(tri,x(:,1),x(:,2),x(:,3),u(:,s));
        shading interp; view([-32 22]);
        caxis([cmin cmax]);
        axis equal; axis off;
        lighting gouraud; camlight left;
        title(['t = ' num2str(t(s)) ', T = ' num2str(T)]);
        pause(0.01);
    end
    fprintf(1,repmat('\b',1,msg_length));
    msg = [num2str(k,'%d') '/' num2str(round(T/dt),'%d')];
    fprintf(1,msg);
    msg_length=numel(msg);
end
fprintf(1,'\n');
disp('-- Simulation completed --------------------------');