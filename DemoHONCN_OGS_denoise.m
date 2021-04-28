% HTVp_OGSTV for Poisson Noise removal
% written by KS Jon, 20200426
% 9.7.0.1190202 (R2019b)
close all;
clc;
clear variables
ima_dir = 'Test images';
maxValuelist    = [350 300 200 100];
etaValueList    = [25 20 15 10];
psf             = fspecial('average', 1); % For denoising. (K is the identity operator)
bgd = 0;

cur_file = 'buty(256).png';
display(sprintf('denoise processing of %s...', cur_file));
for mm = 1:size(maxValuelist, 2)
    
    MaxValue = maxValuelist(mm);
    
    %%
    params = ParamSet(MaxValue);
    params.psf          = fspecial('average', 1); % For denoising. (K is the identity operator)

    params.eta = etaValueList(mm);  % regularization parameters
    
    Img = imread(strcat(ima_dir, filesep, cur_file)); %gray-scale image
    
    H = BlurMatrix(psf,size(Img));
    params.H = H;
    Img = double(Img);
    Img = MaxValue * Img / max(Img(:));
    params.Img = Img;
    % Add Poisson noise to the blurred image
    % the following two lines are to fix the noise value
    stream = RandStream('mt19937ar', 'Seed', 88);
    RandStream.setGlobalStream(stream);
    
    Blr = H * Img + bgd;
    Blr = max(0, Blr);
    Bn = poissrnd(Blr);
    
    % get PSNR and SSIM values for degraded image
    psnr_noisy = psnr(Bn, Img, MaxValue);
    ssim_noisy = ssim(Bn, Img, 'DynamicRange', MaxValue);
    
    tic
    out = HTVp_OGSTV(Bn, params);
    toc
    
    %display result%
    figure, imshow(Bn,[]), title('observed image')
    figure, imshow(out.sol,[ ]), title('recovered image')
    % get PSNR and SSIM values for estimated image
    psnr_est = psnr(out.sol, Img, MaxValue);
    ssim_est = ssim(out.sol, Img, 'DynamicRange', MaxValue);
    
    display(sprintf('noise_level=%d,psnr_noisy=%.2f,ssim_noisy=%.3f,psnr_est=%.2f,ssim_est=%.3f', ...
        MaxValue, psnr_noisy, ssim_noisy, psnr_est, ssim_est));
end
function params = ParamSet(MaxValue)

params.grpSz        = 3; % OGS group size
params.Nit          = 50;
params.Nit_inner    = 5;
params.tol          = 1.0e-3;
params.p            = .1;
params.stepLength   = 1;
params.alpha        = .3;
params.lam          = 3 * MaxValue; 
params.delta         = [0.005,0.03,0.002];

params.MaxValue = MaxValue;

end
