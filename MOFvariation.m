function MOF
clc
radii= 600:100:800;% circle radii [pixels]
p=ceil(2*pi*radii); % approximate perimeter [pixels]
p(mod(p,2)==1)= p(mod(p,2)==1)-1; % ensuring even number
fsigm = @(param,xval) param(1)+(param(2)-param(1))./(1+10.^((param(3)-xval)*param(4)));

for i=1:length(p)
    fprintf('Analyzing circle of radius %5.5f',radii(i))
    fx=ones(1,p(i))*radii(i); %perfect circle of radii(i)
    t= 1:p(i);
    x= linspace(0.00000001,2*pi,p(i));% delta theta constant
    mag=1; % magnitude of the sinusoidal signal
    w0=(2*pi)*(5:20:p(i)/2-50)./p(i); % frequency of the sinusoidal signal
    fx_nx=repmat(fx',1,length(w0))+mag*cos(t'*w0); %radii + sinusoidal signal
    %signal to contour
    [xr,yr]=pol2cart(repmat(x',1,length(w0)),fx_nx);
    m=max(round(2.5*max(abs(xr))));
    n=max(round(2.5*max(abs(yr))));
    xx=xr+repmat(m./2,size(xr));yy=yr+repmat(n./2,size(yr));
    %%%% synthesizing image
    bw=zeros(m,n,length(w0));
    mag_rec=zeros(2,length(w0));
    fprintf('\n')
    fprintf(1,'%d points, progress %3d%%',length(w0),1)
    for ji=1:length(w0)
        fprintf(1,'\b\b\b%3.0f%',ji);
        bw = poly2mask(xx(:,ji),yy(:,ji),m,n);
        s  = regionprops(bw, 'centroid');
        l=bwboundaries(bw );
        centroids = cat(1, s.Centroid);
        l=l{:};
        rhon=NNVI(l,centroids);
        p2=length(rhon);
        if mod(p2,2)
            p2=p2-1;
        end
        Fw = fft(rhon,p2)/length(rhon);
        mag_image=2*abs(Fw(1:p2/2+1));
        [pks,locs] = findpeaks(mag_image(2:end)); %% find max peak
        [~,maxi]=max(pks); %% find max peak
        mag_rec(1,ji)=locs(maxi)+1;
        mag_rec(2,ji)=pks(maxi);
    end
    figure('color',[1 1 1]), plot(1:size(mag_rec,2), mag_rec(2,:)), hold on
    [param,stat]=sigm_fit(1:size(mag_rec,2),mag_rec(2,:));
    mag_sigmoid=fsigm(param,1:size(mag_rec,2));
    plot(1:size(mag_rec,2),mag_sigmoid,'r')
    title(['Magnitude of the circle of radius ',num2str(radii(i))])
    xlabel('Hz')
    ylabel('Magnitude of sinusoidal signal')
    
end
end

function rhon=NNVI(l,centroids)
%Nearest Neighbor Value Interpolation
np=length(l);
l(:,2)=l(:,2)-centroids(1);
l(:,1)=l(:,1)-centroids(2);
[theta,rho] = cart2pol(l(:,2),l(:,1));
dth=linspace(-pi,pi,np);
rhon=zeros(size(dth));

for jj=1:np
    resta=abs(theta-dth(jj));
    [~,kl]=min(resta);
    resta(kl)=100;
    [~,kl2]=min(resta);
    resta(kl2)=100;
    [~,kl3]=min(resta);
    rhon(jj)=mean([rho(kl),rho(kl2),rho(kl3)]);
end
end




function [param,stat]=sigm_fit(x,y,fixed_params,initial_params,plot_flag)
% Optimization of parameters of the sigmoid function
%
% Syntax:
%       [param]=sigm_fit(x,y)
%
%       that is the same that
%       [param]=sigm_fit(x,y,[],[],[])     % no fixed_params, automatic initial_params
%
%       [param]=sigm_fit(x,y,fixed_params)        % automatic initial_params
%       [param]=sigm_fit(x,y,[],initial_params)   % use it when the estimation is poor
%       [param]=sigm_fit(x,y,fixed_params,initial_params,plot_flag)
%
% param = [min, max, x50, slope]
%
% if fixed_params=[NaN, NaN , NaN , NaN]        % or fixed_params=[]
% optimization of "min", "max", "x50" and "slope" (default)
%
% if fixed_params=[0, 1 , NaN , NaN]
% optimization of x50 and slope of a sigmoid of ranging from 0 to 1
%
%
% Additional information in the second output, STAT
% [param,stat]=sigm_fit(x,y,fixed_params,initial_params,plot_flag)
%
%
% Example:
% %% generate data vectors (x and y)
% fsigm = @(param,xval) param(1)+(param(2)-param(1))./(1+10.^((param(3)-xval)*param(4)))
% param=[0 1 5 1];  % "min", "max", "x50", "slope"
% x=0:0.1:10;
% y=fsigm(param,x) + 0.1*randn(size(x));
%
% %% standard parameter estimation
% [estimated_params]=sigm_fit(x,y)
%
% %% parameter estimation with forced 0.5 fixed min
% [estimated_params]=sigm_fit(x,y,[0.5 NaN NaN NaN])
%
% %% parameter estimation without plotting
% [estimated_params]=sigm_fit(x,y,[],[],0)
%
%
% Doubts, bugs: rpavao@gmail.com
% Downloaded from http://www.mathworks.com/matlabcentral/fileexchange/42641-sigmoid-logistic-curve-fit

% warning off

x=x(:);
y=y(:);

if nargin<=1 %fail
    fprintf('');
    help sigm_fit
    return
end

automatic_initial_params=[quantile(y,0.05) quantile(y,0.95) NaN 1];
if sum(y==quantile(y,0.5))==0
    temp=x(y==quantile(y(2:end),0.5));
else
    temp=x(y==quantile(y,0.5));
end
automatic_initial_params(3)=temp(1);

if nargin==2 %simplest valid input
    fixed_params=NaN(1,4);
    initial_params=automatic_initial_params;
    plot_flag=1;
end
if nargin==3
    initial_params=automatic_initial_params;
    plot_flag=1;
end
if nargin==4
    plot_flag=1;
end

if exist('fixed_params','var')
    if isempty(fixed_params)
        fixed_params=NaN(1,4);
    end
end
if exist('initial_params','var')
    if isempty(initial_params)
        initial_params=automatic_initial_params;
    end
end
if exist('plot_flag','var')
    if isempty(plot_flag)
        plot_flag=1;
    end
end

%p(1)=min; p(2)=max-min; p(3)=x50; p(4)=slope como em Y=Bottom + (Top-Bottom)/(1+10^((LogEC50-X)*HillSlope))
%f = @(p,x) p(1) + (p(2)-p(1)) ./ (1 + 10.^((p(3)-x)*p(4)));

f_str='f = @(param,xval)';
free_param_count=0;
bool_vec=NaN(1,4);
for i=1:4;
    if isnan(fixed_params(i))
        free_param_count=free_param_count+1;
        f_str=[f_str ' param(' num2str(free_param_count) ')'];
        bool_vec(i)=1;
    else
        f_str=[f_str ' ' num2str(fixed_params(i))];
        bool_vec(i)=0;
    end
    if i==1; f_str=[f_str ' + (']; end
    if i==2;
        if isnan(fixed_params(1))
            f_str=[f_str '-param(1) )./ (   1 + 10.^( ('];
        else
            f_str=[f_str '-' num2str(fixed_params(1)) ')./ (1 + 10.^(('];
        end
    end
    if i==3; f_str=[f_str ' - xval ) *']; end
    if i==4; f_str=[f_str ' )   );']; end
end

eval(f_str)

[BETA,RESID,J,COVB,MSE] = nlinfit(x,y,f,initial_params(bool_vec==1));
stat.param=BETA';

% confidence interval of the parameters
stat.paramCI = nlparci(BETA,RESID,'Jacobian',J);

% confidence interval of the estimation
[stat.ypred,delta] = nlpredci(f,x,BETA,RESID,'Covar',COVB);
stat.ypredlowerCI = stat.ypred - delta;
stat.ypredupperCI = stat.ypred + delta;

% plot(x,y,'ko') % observed data
% hold on
% plot(x,ypred,'k','LineWidth',2)
% plot(x,[lower,upper],'r--','LineWidth',1.5)

free_param_count=0;
for i=1:4;
    if isnan(fixed_params(i))
        free_param_count=free_param_count+1;
        param(i)=BETA(free_param_count);
    else
        param(i)=fixed_params(i);
    end
end

if plot_flag==1
    x_vector=min(x):(max(x)-min(x))/100:max(x);
    plot(x,y,'k.',x_vector,f(param(isnan(fixed_params)),x_vector),'r-')
    xlim([min(x) max(x)])
end
end