clc;
clear all;
close all;

data=xlsread('cancer.xlsx');

n=size(data,1);
m=size(data,2);

n1=m-1;
n2=3;
n3=1;

eta=0.05;
max_epoch=100;
a=-1;
b=1;
a_l=-1;
b_l=0;
a_u=0;
b_u=1;
alpha=.5;
beta=.5;

rate_train=0.75;
num_train=round(rate_train*n);
num_test=n-num_train;

data_train=data(1:num_train,:);
data_test=data(num_train+1:end,:);

w1_lower=unifrnd(a_l,b_l,[n2 n1]);
net1_lower=zeros(n2,1);
o1_lower=zeros(n2,1);

w1_upper=unifrnd(a_u,b_u,[n2 n1]);
net1_upper=zeros(n2,1);
o1_upper=zeros(n2,1);

w2=unifrnd(a,b,[n3 n2]);
net2=zeros(n3,1);
o2=zeros(n3,1);

error_train=zeros(num_train,1);
error_test=zeros(num_test,1);

output_train=zeros(num_train,1);
output_test=zeros(num_test,1);

mse_train=zeros(max_epoch,1);
mse_test=zeros(max_epoch,1);


%*************************** 
num_w=2*(n2*n1)+n3*n2;      %Layer1 is Rough
num_parameter=num_w+2;      %all weights+alpha+beta
Wall=zeros(1,num_parameter); 
jacob=zeros(num_train,num_parameter);
I=eye(num_parameter);
meu=0.1;
%******************************


for i=1:max_epoch
    for j=1:num_train
        input=data_train(j,1:n1);
        target=data_train(j,1+n1);
        net1_lower=w1_lower*input';
        o1_lower=logsig(net1_lower);
        net1_upper=w1_upper*input';
        o1_upper=logsig(net1_upper);
              
        if o1_lower >= o1_upper
              o1_upper=o1_lower;
              o1_lower = o1_upper;   
        end
        
        o1 = alpha*o1_lower+ beta*o1_upper;
        
        net2=w2*o1;
        o2=net2;

        error_train(j)=target-o2;
        A_lower=diag(o1_lower.*(1-o1_lower));
        A_upper=diag(o1_upper.*(1-o1_upper));
        
        joc_w1_lower=-1*1*alpha*(w2*A_lower)'*input;
        joc_w1_upper=-1*1*beta*(w2*A_upper)'*input; 
        joc_alpha=-1*1*w2*o1_lower;
        joc_beta=-1*1*w2*o1_upper;
        joc_w2=-1*1*o1';
        
        r_w1_lower=reshape(joc_w1_lower,[1 n2*n1]);
        r_w1_upper=reshape(joc_w1_upper,[1 n2*n1]);
        r_w2=reshape(joc_w2,[1 n3*n2]);
        
        r_all_w=[r_w1_lower r_w1_upper r_w2 joc_alpha joc_beta];  
        jacob(j,:)=r_all_w;
        
    end
    
    meu=0.01*error_train'*error_train;
    
    temp_w1_lower=reshape(w1_lower,[1 n2*n1]);
    temp_w1_upper=reshape(w1_upper,[1 n2*n1]);
    temp_w2=reshape(w2,[1 n3*n2]);

    Wall=[temp_w1_lower temp_w1_upper temp_w2 alpha beta];
    Wall=Wall-(inv(jacob'*jacob+meu*I)*jacob'*error_train)';
    
    temp_w1_lower=Wall(1:n2*n1);
    temp_w1_upper=Wall(n2*n1+1:2*n2*n1);
    temp_w2=Wall(2*n2*n1+1:end-2);
    alpha=Wall(end-1);
    beta=Wall(end);
    
    w1_lower=reshape(temp_w1_lower,[n2 n1]);
    w1_upper=reshape(temp_w1_upper,[n2 n1]);
    w2=reshape(temp_w2,[n3 n2]);
    
    for j=1:num_train
        input=data_train(j,1:n1);
        target=data_train(j,1+n1);
        net1_upper=w1_upper*input';
        net1_lower=w1_lower*input';
        o1_upper=logsig(net1_upper);
        o1_lower=logsig(net1_lower);
        
%     if o1_lower >= o1_upper
%               o1_upper=o1_lower;
%               o1_lower = o1_upper;   
%     end

        o1 = alpha*o1_lower+ beta*o1_upper;

        net2=w2*o1;
        o2=net2;
        if o2 <= .5
            o2=0;
        else
            o2=1;
        end
        output_train(j)=o2;
        error_train(j)=target-o2;
    end
    
    mse_train(i)=mse(error_train);
    
    for j=1:num_test
        input=data_test(j,1:n1);
        target=data_test(j,1+n1);
        net1_upper=w1_upper*input';
        net1_lower=w1_lower*input';
        o1_upper=logsig(net1_upper);
        o1_lower=logsig(net1_lower);
        
%        if o1_lower >= o1_upper
%               o1_upper=o1_lower;
%               o1_lower = o1_upper;   
%        end
%         
%         alpha= alpha-eta*error_train(j)*-1*1*w2*o1_upper;
%         beta= beta-eta*error_train(j)*-1*1*w2*o1_lower;
        o1 = alpha*o1_lower+ beta*o1_upper;
        net2=w2*o1;
        o2=net2;
        if o2 <= .5
            o2=0;
        else
            o2=1;
        end
        output_test(j)=o2;
        error_test(j)=target-o2;
    end
    
    mse_test(i)=mse(error_test);
    
end  
    figure(1);
    subplot(2,2,1),plot(data_train(:,n1+1),'-r');
    hold on;
    subplot(2,2,1),plot(output_train,'-b');
    hold off;
    
    subplot(2,2,3),plot(data_test(:,n1+1),'-r');
    hold on;
    subplot(2,2,3),plot(output_test,'-b');
    hold off;
    
    subplot(2,2,2),semilogy(mse_train(1:i),'-r');
    subplot(2,2,4),semilogy(mse_test(1:i),'-r');

figure(2);
plotregression(data_train(:,n1+1),output_train);

figure(3);
plotregression(data_test(:,n1+1),output_test);

r=1;

mse_train(max_epoch)
mse_test(max_epoch)

figure(4)
target=data_train(:,n1+1);
output=output_train;
d_t1=zeros(num_train,2);
d_t2=zeros(num_train,2);
for i=1:num_train
    if(target(i)==0)
        d_t1(i,:)=[1 0];
    end
    if(target(i)==1)
        d_t1(i,:)=[0 1];
    end
end

for i=1:num_train
    if(output(i)==0)
        d_t2(i,:)=[1 0];
    end
    if(output(i)==1)
        d_t2(i,:)=[0 1];
    end
end

plotconfusion(d_t1',d_t2');

figure(5)
target=data_test(:,n1+1);
output=output_test;
d_t1=zeros(num_test,2);
d_t2=zeros(num_test,2);

for i=1:num_test
    if(target(i)==0)
        d_t1(i,:)=[1 0];
    end
    if(target(i)==1)
        d_t1(i,:)=[0 1];
    end
end

for i=1:num_test
    if(output(i)==0)
        d_t2(i,:)=[1 0];
    end
    if(output(i)==1)
        d_t2(i,:)=[0 1];
    end
end

plotconfusion(d_t1',d_t2');
