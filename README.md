# hello_world
Just another respository
%----------------------------------------------------------------------------
% 自适应神经模糊推理系统(ANFIS, adaptive network-based fuzzy inference system)
%----------------------------------------------------------------------------
% 全年日水平太阳辐射模型(DYB, day of year based)
% 输入: 天数 n_day
%% 定义
clc;
clear;
format compact;
load data_H.mat;
FisNum=3;        %模糊函数个数
EpochN=1500;     %迭代次数
H_PathMax1=1000; %原始日辐射最大差值1
H_PathMax2=50;   %原始日辐射最大差值2
PremRatio=0.0012; %前提参数学习率
ConsRatio=0.0033; %结论参数学习率
%% 数据预处理
%滤波
for i=3:1825
    if H(i)-H(i-1)>H_PathMax1 || H(i)-H(i-1)<-H_PathMax1
        H(i)=H(i-1)*0.5+H(i-2)*0.5;%差值过大舍弃
    elseif H(i)-H(i-1)>H_PathMax2 || H(i)-H(i-1)<-H_PathMax2
        H(i)=H(i)*0.4+H(i-1)*0.3+H(i-2)*0.3;%差值较大滤波
    end
end
%初始化训练集
X=(1:365)';
L=H(X);
[NormX,PsX]=mapminmax(X',0,1);
NormX=NormX';% 0 1
[NormL,PsL]=mapminmax(L',0,1);
NormL=NormL';% 1 0
%% 工具包
% Epoch=EpochN;%迭代次数
% FIS=genfis1([NormX,NormL],2,'gaussmf');%生成模糊推理系统
% ANFIS=anfis([NormX,NormL],FIS,Epoch);  %建立模型
% OutRight=evalfis(NormX,ANFIS); %运用模型计算结果
%% 参数初始化
%前提参数(高斯) sig c
PremPara=rand(FisNum,2)%随机初值
% PremPara=[ 0.3708    0.8235;
%            0.2046    0.9574;
%            0.6583    0.1947];
% 0.2465    1.1483
% 0.3408    0.9378
% 0.6118   -0.2647
%结论参数 p q r
ConsPara=rand(FisNum,3)%随机初值
% ConsPara=[ 0.3278    0.0361    0.1817;
%            0.0840    0.3591    0.7584;
%            0.9528    0.5810    0.1938];
% -0.1222    0.0361    0.3582
% -0.7197    0.3591    0.4289
%  2.2045    0.5810    0.0249
%% 模型训练与参数更新
ReFlag=0;
dPremParaSum=zeros(FisNum,2);
for epoch=1:EpochN
    for i=1:max(X)
        x=NormX(i);
        y=0;%单输入未使用
        %第一层-隶属度函数模糊化
        for j=1:FisNum
        A(j)=gaussmf(x,PremPara(j,:));%钟形
        end
        %第二层-激励强度计算(模糊集代数乘积)
        W=A';
        %第三层-激励强度归一化
        W_Norm=W./norm(W);
        %第四层-规则的输出
        O4=W_Norm.*([x,y,1]*ConsPara')';
        %第五层-输出层，结论参数的调整
        O5=sum(O4);
        Delta5=NormL(i)-O5;
        d(epoch,i)=abs(Delta5);
        Out(epoch,i)=O5;
        dConsPara=Delta5*W_Norm*[x,y,1];
        ConsPara=ConsPara+ConsRatio*dConsPara;
        %反向传播求前提参数
        Delta4=Delta5*([x,y,1]*ConsPara')';
        Delta3=Delta4.*([x,y,1]*ConsPara')';%??
        Delta2=Delta3.*((W.*0+sum(W)-W)./(sum(W))^2);
        for j=1:FisNum
            sig=PremPara(j,1);c=PremPara(j,2);
            dPremPara(j,1)= Delta2(j)*W(j)*(x-c)/sig/sig;
            dPremPara(j,2)= Delta2(j)*W(j)*((x-c)^2)*sig^(-3);
        end
        ReFlag=ReFlag+1;
        dPremParaSum=dPremParaSum+dPremPara;
        if ReFlag>=8
            PremPara=PremPara+PremRatio*dPremParaSum/8;
            ReFlag=0;
            dPremParaSum=dPremParaSum.*0;
        end
    end
    %误差显示
    if mod(epoch,100)==0
        disp(['迭代',num2str(epoch),'次 ','误差： ',num2str(mean(abs(d(epoch,1:365))))]);
    end
    %------------------结果输出---------------------
    if epoch==EpochN
        for i=1:max(X)
            x=NormX(i);
            y=0;%单输入未使用
            %第一层-隶属度函数模糊化
            for j=1:FisNum
                A(j)=gaussmf(x,PremPara(j,:));%钟形
            end
            %第二层-激励强度计算(模糊集代数乘积)
            W=A';
            %第三层-激励强度归一化
            W_Norm=W./norm(W);
            %第四层-规则的输出
            O4=W_Norm.*([x,y,1]*ConsPara')';
            %第五层-输出层，结论参数的调整
            Out(EpochN+1,i)=sum(O4);
        end
    end
end
%图形显示
plot(X,NormL,'b');
hold on;
plot(X,Out(epoch+1,:),'r');
% plot(X,OutRight,'k');
