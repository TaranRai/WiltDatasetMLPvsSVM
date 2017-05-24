% SVM
clear; clc;

% set parameters
pBoxConstraint = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0];
pRadialBasis   = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0];

% read in data from files
TrainingFile = 'SMOTEtraining4Boosted.csv'; % 'TrainingSmote5.csv';
TestingFile  = 'SMOTEtesting4Boosted.csv'; %'Testing.csv';
Tr = csvread(TrainingFile);
Te = csvread(TestingFile);

Features = Tr(:,1:4);
Targets = Tr(:,5);
%Targets(Targets==-1)=0;

%[coeff,score,latent,tsquared,explained] = pca(Features);

LowestParams = [0.0 0.0 1.0];
% Grid search on training data
% Start timer
t0 = cputime;
for x=1:size(pBoxConstraint,2)
    for y=1:size(pRadialBasis,2)
        %SVMModel = fitclinear(Features(:,1:3), Targets);
        SVMModel = fitcsvm(Features, Targets, 'Standardize', true, ...                                              
                                              'KernelFunction','RBF', ...
                                              'KernelScale', pRadialBasis(y), ...
                                              'BoxConstraint', pBoxConstraint(x) ...
                                              );
        
        % calculate cross-validation error
        CVSVMModel = crossval(SVMModel, 'KFold', 3);
        classLoss = kfoldLoss(CVSVMModel);
        
        LowestParamFlag = false;
        if classLoss < LowestParams(3)
            LowestParams = [pBoxConstraint(x) pRadialBasis(y) classLoss];
            LowestParamFlag = true;
        end
        
        %LogicalStr = {'false', 'true'};
        display(['BoxConstraint: ' num2str(pBoxConstraint(x)) ...
                 ', KernelScale: ' num2str(pRadialBasis(y)) ...
                 ', CVSV Error: ' num2str(classLoss) ...
                 ', LowestErrorFlag: ' num2str(LowestParamFlag)
                 ])        
    end
end
%LowestParams = [0.95 0.3 0.0];
display(['Grid Search Time: ' num2str(cputime-t0)])

% Apply optimal parameters to new training model
% Start timer
t1 = cputime;
%SVMModel = fitcsvm(Features, Targets, 'Standardize', true);
SVMModel = fitcsvm(Features, Targets, 'Standardize', true, ...                                              
                                      'KernelFunction','RBF', ...
                                      'KernelScale', LowestParams(1), ...
                                      'BoxConstraint', LowestParams(2) ...
                                      );                                  
display(['SVM Training Time: ' num2str(cputime-t1)])
                                  
% Calculate prediction scores/error/accuracy
mdlSVM = fitPosterior(SVMModel);
[~,score_svm] = resubPredict(mdlSVM);
%[~,score_svm] = predict(mdlSVM, mdlSVM.X);
[~, indexes] = max(score_svm,[],2);
indexes = indexes-1;
compare = [Targets indexes];
predictionError = sum(abs(diff(compare,1,2))) / size(compare,1);
display(['Prediction Error: ' num2str(predictionError)])

% Apply new model, with optimal parameters, to testing data
TeFeatures = Te(:,1:4);
TeTargets = Te(:,5);
%TeTargets(TeTargets==-1)=0;
[isLabels_SVM, score_SVM] = predict(SVMModel, TeFeatures);

% Calculate confustion matrix
SVM_ConfusionMat = confusionmat(TeTargets, isLabels_SVM)

% Calculate precision and recall
%display(['Recall - SVM:' num2str(recall(SVM_ConfusionMat)')])
%display(['Precision - SVM:' num2str(precision(SVM_ConfusionMat)')])

% draw confusion matrix
figure
DrawCharts(5,SVM_ConfusionMat, 'SVM');

%load ionosphere
%  resp = Targets;   %strcmp(Y,'b'); % resp = 1, if Y = 'b', or 0 if Y = 'g' 
%  pred = Features;    %X(:,3:34);
%  mdlSVM = fitcsvm(pred,resp,'Standardize',true);
%  mdlSVM = fitPosterior(mdlSVM);
%  [~,score_svm] = resubPredict(mdlSVM);
% 
%  a = [0;1];
%  L = logical(a);
%  
%  NewTargets = Te(1:500,6);
%  NewTargets(NewTargets==-1)=0;
%  
%  [Xsvm,Ysvm,Tsvm,AUCsvm] = perfcurve(NewTargets,score_svm(:,L),0);
%  plot(Xsvm,Ysvm)
%  legend('Logistic Regression','Support Vector Machines','Naive Bayes','Location','Best')
%  xlabel('False positive rate'); ylabel('True positive rate');
%  title('ROC Curves for Logistic Regression, SVM, and Naive Bayes Classification')

% Figure 1
% sv = SVMModel.SupportVectors;
% figure
% gscatter(Features(:,1),Features(:,2),Targets)
% hold on
% plot(sv(:,1),sv(:,2),'ko','MarkerSize',10)
% legend('Feature A','Feature B','Support Vector')
% hold off
% 
% [~,~,id] = unique(Targets);
% colors = 'rgb';
% markers = 'osd';
% 
% figure
% for idx = 1 : 3
%     data = Features(id == idx,:);
%     plot3(Features(:,1), Features(:,2), Features(:,3), [colors(idx) markers(idx)]);
%     hold on;
% end
% grid;
% hold off
% 
% 
% % Figure 2
% rng(1);
% SVMModel = fitcsvm(Features,Targets,'KernelScale','auto','Standardize',true,'OutlierFraction',0.05);
% svInd = SVMModel.IsSupportVector;
% h = 50 %0.02; % Mesh grid step size
% [X1,X2] = meshgrid(min(Features(:,1)):h:max(Features(:,1)),...
%     min(Features(:,2)):h:max(Features(:,2)));
% [~,score] = predict(SVMModel,[X1(:),X2(:)]);
% scoreGrid = reshape(score,size(X1,1),size(X2,2));
% 
% figure
% plot(Features(:,1),Features(:,2),'k.')
% hold on
% plot(Features(svInd,1),Features(svInd,2),'ro','MarkerSize',10)
% contour(X1,X2,scoreGrid)
% colorbar;
% title('{\bf Iris Outlier Detection via One-Class SVM}')
% xlabel('Sepal Length (cm)')
% ylabel('Sepal Width (cm)')
% legend('Observation','Support Vector')
% hold off
% 
% % Figure 3
% figure
% gscatter(X(:,1),X(:,2),Y);
% h = gca;
% lims = [h.XLim h.YLim]; % Extract the x and y axis limits
% title('{\bf Scatter Diagram of Iris Measurements}');
% xlabel('Petal Length (cm)');
% ylabel('Petal Width (cm)');
% legend('Location','Northwest');
% 
% % Figure 4
% %Color in the regions of the plot based on which class the corresponding new observation belongs.
% figure
% h(1:3) = gscatter(xGrid(:,1),xGrid(:,2),maxScore,...
%     [0.1 0.5 0.5; 0.5 0.1 0.5; 0.5 0.5 0.1]);
% hold on
% h(4:6) = gscatter(X(:,1),X(:,2),Y);
% title('{\bf Iris Classification Regions}');
% xlabel('Petal Length (cm)');
% ylabel('Petal Width (cm)');
% legend(h,{'setosa region','versicolor region','virginica region',...
%     'observed setosa','observed versicolor','observed virginica'},...
%     'Location','Northwest');
% axis tight
% hold off
