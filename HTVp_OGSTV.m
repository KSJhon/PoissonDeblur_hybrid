function out = HTVp_OGSTV(Img, opts)

%%%.
% Overlapping Group Sparse and Nonconvex Second-order Total Variation Priors for Restoring Poisson Noisy Images
%

f               = Img;
[row, col] 		= size(Img);

grpSz           = opts.grpSz; %Group size
Nit             = opts.Nit;
Nit_inner       = opts.Nit_inner;
tol             = opts.tol;
lam             = opts.lam; % The regularization parameter.
delta           = opts.delta;
eta             = opts.eta;
alpha           = opts.alpha;
stepLength      = opts.stepLength;
p               = opts.p;
relError        = zeros(Nit,1); % Compute error relative to the previous iteration.

psf = opts.psf;
H = opts.H;
Maxvalue = opts.MaxValue;

%**************Initialize Lagrange Multipliers***********************

% Two Lagrange multipliers for the x2 sub-problems (The OGS term)

omega0        = zeros(row, col); % Multiplier for Hf

omega1        = zeros(row, col); % Multiplier for Dux
omega2        = zeros(row, col); % Multiplier for Duy

% Four Lagrange multipliers for the w sub-problems (The 2nd order
% non-convex term)

omega3        = zeros(row, col); %Multiplier for Duxx
omega4        = omega3;            %Multiplier for Duxy
omega5        = omega3;            %Multiplier for Duyx
omega6        = omega3;            %%Multiplier for Duyy

%*************** x2 sub-problem variable initialization ******************

x21         = zeros(row, col); % x solution of the x2 sub-problem for Dux
x22         = x21; % y solutiono fhte x2 sub-problem for Duy

%************** x3 sub-problem variable initialization *******************

x31         = zeros(row, col);
x32         = x31;
x33         = x31;
x34         = x31;

otf1  = psf2otf([1, -1],  [row col]);
otf2  = psf2otf([1; -1],  [row col]);
otf3 = psf2otf([1 -2 1], [row col]);
otf4 = psf2otf([1 -1;-1 1], [row col]);
otf5 = psf2otf([1 -1;-1 1], [row col]);
otf6 = psf2otf([1; -2; 1], [row col]);

eigK            = psf2otf(psf, [row col]); %In the fourier domain
eigKtK          = abs(eigK).^2;

eigDtD  = abs(otf1).^2 + abs(otf2).^2;
eigDDtDD = abs(otf3).^2 + abs(otf4).^2 ...
            + abs(otf5).^2 + abs(otf6).^2;
lhs  = delta(1) * eigKtK + delta(2) * eigDtD + delta(3) * eigDDtDD; %hybrid + 1
[D, Dt]      = defDDt(); %Declare forward finite difference operators
[DD, DDt]    = defDDt2;  %Declare 2nd order forward finite difference operators

[Dux, Duy] = D(f);
[Duxx, Duxy, Duyx, Duyy] = DD(f);

curNorm = sqrt(norm(Duxx(:) - x31(:), 'fro'))^2 + sqrt(norm(Duxy(:) - x32(:), 'fro'))^2 + ...
    sqrt(norm(Duyx(:) - x33(:),'fro'))^2 + sqrt(norm(Duyy(:) - x34(:),'fro'))^2;

for k = 1:Nit
    % x1 sub-problem
    Hf = H * f;
    temp = Hf - omega0 - lam / delta(1);
    x1 = 0.5 * (temp + sqrt(temp.^2 + 4 * lam .* Img / delta(1)));
    
    %********* f sub-problem (Least squares problem, use FFT's)********
    f_old   = f;

    rhs = delta(1) * conj(eigK).*fft2(x1) + delta(1) * conj(eigK).*fft2(omega0) +...
            delta(2) * conj(otf1).*fft2(x21) + delta(2) * conj(otf1).*fft2(omega1) +...
            delta(2) * conj(otf2).*fft2(x22) + delta(2) * conj(otf2).*fft2(omega2) +...
            delta(3) * conj(otf3).*fft2(x31) + conj(otf3).*fft2(omega3) +...
            delta(3) * conj(otf4).*fft2(x32) + conj(otf4).*fft2(omega4) +...
            delta(3) * conj(otf5).*fft2(x33) + conj(otf5).*fft2(omega5) +...
            delta(3) * conj(otf6).*fft2(x34) + conj(otf6).*fft2(omega6);
    f       = rhs./lhs;

    f       = real(ifft2(f));

    [Dux, Duy] = D(f);
    [Duxx, Duxy, Duyx, Duyy] = DD(f);
    
    %***************** x2 sub-problem (Group sparse problem)***********
    x21 = gstvdm(Dux - omega1 , grpSz , 1/delta(2), Nit_inner);
    x22 = gstvdm(Duy - omega2 , grpSz , 1/delta(2), Nit_inner);
    
    %***** x3 sub-problem (2nd order non-convex problem, IRLS algo)**
    
    q1      = Duxx - omega3/delta(3);
    q2      = Duxy - omega4/delta(3);
    q3      = Duyx - omega5/delta(3);
    q4      = Duyy - omega6/delta(3);
    
    wgt1 = eta * p./(Duxx.^2 + eps).^(1 - p/2); %IRLS Weight update 
    wgt2 = eta * p./(Duxy.^2 + eps).^(1 - p/2);% IRLS Weight update
    wgt3 = eta * p./(Duyx.^2 + eps).^(1 - p/2);% IRLS Weight update
    wgt4 = eta * p./(Duyy.^2 + eps).^(1 - p/2);% IRLS Weight update
    
    x31      = shrink(q1, wgt1./delta(3));
    x32      = shrink(q2, wgt2./delta(3));
    x33      = shrink(q3, wgt3./delta(3));
    x34      = shrink(q4, wgt4./delta(3));
    
    
    %******* omega update (Lagrange multiplier update)******************
    
    omega0     = omega0 + (x1 - Hf);
    omega1     = omega1 + (x21 - Dux);
    omega2     = omega2 + (x22 - Duy);
    
    omega3     = omega3 + delta(3) * (x31 - Duxx);
    omega4     = omega4 + delta(3) * (x32 - Duxy);
    omega5     = omega5 + delta(3) * (x33 - Duyx);
    omega6     = omega6 + delta(3) * (x34 - Duyy);
    
    %***** Some statistics ***
    relError(k)    = norm(f - f_old, 'fro')/norm(f, 'fro');
    
    if relError(k) < tol
        break;
    end
       
    normOld = curNorm;
    curNorm = sqrt(norm(Duxx(:) - x31(:), 'fro'))^2 + sqrt(norm(Duxy(:) - x32(:), 'fro'))^2 + ...
        sqrt(norm(Duyx(:) - x33(:), 'fro'))^2 + sqrt(norm(Duyy(:) - x34(:), 'fro'))^2;
    
    if curNorm > alpha * normOld
        delta = stepLength * delta;
    end
 
end
out.sol                 = f;
out.relativeError       = relError(1:k);
out.OverallItration     = size(out.relativeError,2); %No of itr to converge
end


function [D,Dt] = defDDt()
D  = @(x1) ForwardDiff(x1);
Dt = @(X,Y) Dive(X,Y);
end

function [Dux,Duy] = ForwardDiff(x1)
Dux = [diff(x1,1,2), x1(:,1,:) - x1(:,end,:)];
Duy = [diff(x1,1,1); x1(1,:,:) - x1(end,:,:)];
end

function DtXY = Dive(X,Y)
% Transpose of the forward finite difference operator
% is the divergence fo the forward finite difference operator
DtXY = [X(:,end) - X(:, 1), -diff(X,1,2)];
DtXY = DtXY + [Y(end,:) - Y(1, :); -diff(Y,1,1)];
end

function [DD,DDt] = defDDt2
% defines finite difference operator D
% and its transpose operator
DD  = @(x1) ForwardD2(x1);
DDt = @(Duxx,Duxy,Duyx,Duyy) Dive2(Duxx,Duxy,Duyx,Duyy);
end

function [Duxx Duxy Duyx Duyy] = ForwardD2(x1)
%
Duxx = [x1(:,end) - 2*x1(:,1) + x1(:,2), diff(x1,2,2), x1(:,end - 1) - 2*x1(:,end) + x1(:,1)];
Duyy = [x1(end,:) - 2*x1(1,:) + x1(2,:); diff(x1,2,1); x1(end - 1,:) - 2*x1(end,:) + x1(1,:)];
%
Aforward = x1(1:end - 1, 1:end - 1) - x1(  2:end,1:end - 1) - x1(1:end - 1,2:end) + x1(2:end,2:end);
Bforward = x1(    end, 1:end - 1) - x1(      1,1:end - 1) - x1(    end,2:end) + x1(    1,2:end);
Cforward = x1(1:end - 1,     end) - x1(1:end - 1,      1) - x1(  2:end,  end) + x1(2:end,    1);
Dforward = x1(    end,     end) - x1(      1,    end) - x1(    end,    1) + x1(    1,    1);
%
Eforward = [Aforward ; Bforward]; Fforward = [Cforward ; Dforward];
Duxy = [Eforward, Fforward]; Duyx = Duxy;
%
end

function Dt2XY = Dive2(Duxx, Duxy, Duyx, Duyy)
%
Dt2XY =         [Duxx(:,end) - 2 * Duxx(:, 1) + Duxx(:, 2), diff(Duxx, 2, 2), Duxx(:, end - 1) - 2 * Duxx(:, end) + Duxx(:, 1)]; % xx
Dt2XY = Dt2XY + [Duyy(end, :) - 2 * Duyy(1, :) + Duyy(2, :); diff(Duyy, 2, 1); Duyy(end - 1, :) - 2 * Duyy(end, :) + Duyy(1, :)]; % yy
%
Axy = Duxy(1    ,    1) - Duxy(      1,    end) - Duxy(    end,    1) + Duxy(    end,    end);
Bxy = Duxy(1    , 2:end) - Duxy(      1, 1:end - 1) - Duxy(    end, 2:end) + Duxy(    end, 1:end - 1);
Cxy = Duxy(2:end,    1) - Duxy(1:end - 1,      1) - Duxy(  2:end,  end) + Duxy(1:end - 1,    end);
Dxy = Duxy(2:end, 2:end) - Duxy(  2:end, 1:end - 1) - Duxy(1:end - 1, 2:end) + Duxy(1:end - 1, 1:end - 1);
Exy = [Axy, Bxy]; Fxy = [Cxy, Dxy];
%
Dt2XY = Dt2XY + [Exy; Fxy];
Dt2XY = Dt2XY + [Exy; Fxy];
end

function z = shrink(x, r)
z = x ./ (1 + r);
end

