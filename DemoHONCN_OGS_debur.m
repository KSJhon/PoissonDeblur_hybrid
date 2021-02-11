% HTVp_OGSTV for Poisson non-blind deblurring
% written by KS Jon, 20200426

close all;
clc;
clear variables
ima_dir = 'Test images';
deblurringCase = 2;
switch deblurringCase
    case 1 %Gaussin deblurring
        psf= fspecial('gaussian', [9 9], 1);
        etaValueList  = [18 14 6 2];
        DeblurSetting = 'G(9,1)';
    case 2 %motion deblurring
        psf = fspecial('motion', 5, 45);
        etaValueList  = [8 6 4 1];
        DeblurSetting = 'M(5,45)';
    case 3 %ground truth deblurring
        psf = load('im01_ker05.mat');
        psf = psf.f;
        etaValueList  = [4 3 2 .2];
        DeblurSetting = 'GndTrth';
    otherwise
        disp('other setting')
end
maxValuelist    = [350 300 200 100];
bgd = 0;

cur_file = 'buty(256).png';
display(sprintf('%s deblur processing of %s...', DeblurSetting,cur_file));

for mm = 1:size(maxValuelist, 2)
    
    MaxValue = maxValuelist(mm);
    params = ParamSet(MaxValue);    %load default parameters
    params.psf = psf;
    params.eta = etaValueList(mm);  % regularization parameter
    Img = imread(strcat(ima_dir, filesep, cur_file)); %gray-scale image
    
    H = BlurMatrix(psf,size(Img));  % blur matrix
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
    psnr_blurry = psnr(Bn, Img, MaxValue);
    ssim_blurry = ssim(Bn,Img,'DynamicRange', MaxValue);
    
    tic
    out = HTVp_OGSTV(Bn, params);
    toc
    
    %display%
    figure, imshow(Bn,[]), title('observed image')
    figure,imshow(out.sol,[]), title('recovered image')
    
    % get PSNR and SSIM values for estimated image
    psnr_est = psnr(out.sol, Img, MaxValue);
    ssim_est = ssim(out.sol,Img,'DynamicRange', MaxValue);
    
    display(sprintf('DeblurSetting=%s,noise_level=%d,psnr_blurry=%.2f,ssim_blurry=%.3f,psnr_est=%.2f,ssim_est=%.3f', ...
        DeblurSetting, MaxValue, psnr_blurry, ssim_blurry, psnr_est, ssim_est));
    
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
params.delta         = [0.01,0.1,0.01];

params.MaxValue = MaxValue;

end
