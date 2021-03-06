clear; clc;
% import training dataset
TrainingFile = 'SMOTEtraining4Boosted.csv';
Tr = csvread(TrainingFile);
Features = Tr(:, 1:4); 
Class = Tr(:, 5);

% import testing dataset
TestingFile = 'SMOTEtesting4Boosted.csv';
Te = csvread(TestingFile);
TeFeatures = Te(:,1:4);
TeClass = Te(:,5);

hiddenLayer = [5, 10, 15, 20, 25, 30];
epochs = [50, 100, 150, 200, 250, 300];
learningRate = [0.1, 0.2, 0.5, 0.7, 0.9];
momentum = [0.1, 0.3, 0.5, 0.7, 0.9];

LowestParams = [0.0 0.0 0.0 0.0 1.0];

% Start timer
t0 = cputime;

for a = 1:length(hiddenLayer)
    for b = 1:length(epochs)
        for c = 1:length(learningRate)
            for d = 1:length(momentum)
                %display(['hiddenLayer = ' num2str(hiddenLayer(a)) ', epochs = ' num2str(epochs(b)) ',learningRate = ' num2str(learningRate(c))])
                %perf = 0;
                perf = DoGridSearch(hiddenLayer(a), epochs(b), learningRate(c), momentum(d), Features, Class);
                LowestParamFlag = false;
                if perf < LowestParams(length(LowestParams))
                    LowestParams = [ hiddenLayer(a) epochs(b) learningRate(c) momentum(d) perf ];
                    LowestParamFlag = true;
                    display(['low parameter logged hiddenLayer = ' num2str(LowestParams(1)) ', epochs = ' num2str(LowestParams(2)) ', learningRate = ' num2str(LowestParams(3)) ', momentum = ' num2str(LowestParams(4)) ', perf = ' num2str(LowestParams(4))])
                end
            end
        end
    end
end
display(['Grid Search Time: ' num2str(cputime-t0)])

display(['lowest parameters hiddenLayer = ' num2str(LowestParams(1)) ', epochs = ' num2str(LowestParams(2)) ', learningRate = ' num2str(LowestParams(3)) ', momentum = ' num2str(LowestParams(4)) ', perf = ' num2str(LowestParams(4))])

% Start timer
t1 = cputime;
%[Perf, Network] = DoGridSearch(LowestParams(1),LowestParams(2),LowestParams(3),LowestParams(4),Features,Class);
[Perf, Network] = DoGridSearch(20, 300, 0.1, 0.5, Features, Class);
display(['MLP Training Time: ' num2str(cputime-t1)])
display(['Performance on Training Dataset with Best (Lowest) Params: ' num2str(Perf)])
%figure, plotconfusion(Class',Network(Features'))

x = TeFeatures';
t = TeClass';
y = Network(x);
e = gsubtract(t,y);
performance = perform(Network,t,y);
display(['Performance on Testing Dataset: ' num2str(performance)])
figure, ploterrhist(e)
figure, plotconfusion(t,y)
figure, plotroc(t,y)

function [performance, net] = DoGridSearch(a,b,c,d,Features,Class)
% Solve a Pattern Recognition Problem with a Neural Network
% Script generated by Neural Pattern Recognition app
% Created 26-Mar-2017 18:45:46
%
% This script assumes these variables are defined:
%
%   Features - input data.
%   Class - target data.

x = Features';
t = Class';

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'traingdx';  % Scaled conjugate gradient backpropagation.

% Create a Pattern Recognition Network
hiddenLayerSize = a;
net = patternnet(hiddenLayerSize, trainFcn);

%gridsearch parameter one- epochs
net.trainParam.epochs = b;

%gridsearch parameter 2 - learning rate
net.trainParam.lr = c;

%gridsearch parameter 3 - momentum
net.trainParam.mc = d;

% For a list of all performance functions type: help nnperformance
net.performFcn = 'mse';  % mse


% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Train the Network
[net,tr] = train(net,x,t);

% Test the Network
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y);
tind = vec2ind(t);
yind = vec2ind(y);
percentErrors = sum(tind ~= yind)/numel(tind);

% Recalculate Training, Validation and Test Performance
%testTargets = t .* tr.testMask{1};
%testPerformance = perform(net,testTargets,y)


% View the Network
%view(net)

% Plots
% Uncomment these lines to enable various plots.
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, ploterrhist(e)
%figure, plotconfusion(t,y)
%figure, plotroc(t,y)

end