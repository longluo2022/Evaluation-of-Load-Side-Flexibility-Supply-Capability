% Author: icy honey tea
% Date: 2023/10/19
 
clc,clear
poolobj = gcp('nocreate');
delete(poolobj);
setdemorandstream(pi);
%% 数据导入
PV_normal = xlsread('风光场景数据.xlsx','光伏典型场景');
PV_abnormal = xlsread('风光场景数据.xlsx','光伏极端场景');
PWIND_normal = xlsread('风光场景数据.xlsx','风电典型场景');
PWIND_abnormal = xlsread('风光场景数据.xlsx','风电极端场景');
LOAD_normal = xlsread('风光场景数据.xlsx','负荷典型场景');
LOAD_abnormal = xlsread('风光场景数据.xlsx','负荷极端场景');
LOAD_normal = LOAD_normal/8;
LOAD_abnormal = LOAD_abnormal/8;
GEN_data = xlsread('54火电机组数据.xlsx','3机组');
temperature = xlsread('上海2021季节典型日气温.xlsx','B2:E86401');

%% 系统参数
N = size(GEN_data,1);      %发电机数量
T = 48;     %15min/点
Temperature = 30;  %室外温度
PV_normal = PV_normal(:,1:end/T:end);
PV_abnormal = PV_abnormal(:,1:end/T:end);
PWIND_normal = PWIND_normal(:,1:end/T:end);
PWIND_abnormal = PWIND_abnormal(:,1:end/T:end);
LOAD_normal = LOAD_normal(:,1:end/T:end);
LOAD_abnormal = LOAD_abnormal(:,1:end/T:end);
% Pmax = [8000;8000;8000]; % 发电机组出力上限
% Pmin = [200;200;200];   % 发电机组出力下限
% R_up = [1000,1000,1000];  % 向上爬坡速率
% R_down = [1000,1000,1000];  % 向下爬坡速率
% T_start = [2,2,2];  % 最小启动时间
% T_stop = [2,2,2];   % 最小关停时间
Pmax = GEN_data(:,2); % 发电机组出力上限
Pmin = GEN_data(:,3);   % 发电机组出力下限
R_up = GEN_data(:,7)';  % 向上爬坡速率
R_down = GEN_data(:,7)';  % 向下爬坡速率
T_start = GEN_data(:,8)';  % 最小启动时间
T_stop = GEN_data(:,9)';   % 最小关停时间
% 计算成本系数------------
% Ai = diag([0.04 0.01 0.02]);
% Bi = [10 20 20];
% C_start = [40, 25, 5];          %启动成本
% C_stop = [20, 12.50, 2.50];     %关停成本
Ai = diag(GEN_data(:,6));
Bi = GEN_data(:,5)';
C_start = GEN_data(:,10)';          %启动成本
C_stop = GEN_data(:,10)';     %关停成本

%% 数据中心
DAY = 4;
task = xlsread('数据中心负载利用率.xlsx','负载数量'); %工作负载数量
task = task(288*(DAY-1)+1:3:DAY*288,2)';
Num = 4000; %服务器数量
utilization = 0.6*task/max(task)*100;  %模拟利用率及负载数量关系
[P_server,P_chiller,P_aircon,P_pdu,P_ups,P_dc] = datacenter(utilization, Num, Temperature); %数据中心总功率（转移前）
task = task(1:end/T:end);
P_dc = P_dc(1:end/T:end);
x = [1:9/(T-1):10];%不同时段可延迟负载数量占比
pp = -0.003176*x.^4 + 0.07618*x.^3 + -0.6079*x.^2 + 1.717*x + -0.7118;
tp = [0.7018, 0.0929, 0.1135, 0.0918];%各类可延迟负载占比
tt = [15, 60, 120, 240]/(24*60/T);%各类可延迟时限
task_various(1,:) = round(pp.*task*tp(1));%各类可转移负载数量
task_various(2,:) = round(pp.*task*tp(2));
task_various(3,:) = round(pp.*task*tp(3));
task_various(4,:) = round(pp.*task*tp(4));
task_various(2,1) = task_various(2,1)+1;
task_various_sum = sum(sum(task_various));%可转移负载总数量
P_dc_various(1,:) = P_dc.*task_various(1,:)./task;%各类可转移负载对应功率
P_dc_various(2,:) = P_dc.*task_various(2,:)./task;
P_dc_various(3,:) = P_dc.*task_various(3,:)./task;
P_dc_various(4,:) = P_dc.*task_various(4,:)./task;
P_dc_various_sum = sum(sum(P_dc_various));%总可转移负载对应功率
%计算各类可延迟上限
for j = 1:4
    for i = 1:length(task_various(j,:))
        if i-tt(j)<=0
            task_up(j,i) = sum(task_various(j,1:i));
        else
            task_up(j,i) = sum(task_various(j,int8(i-tt(j)):i));
        end
    end
end
P_up = P_dc.*task_up./task;%计算各类可转移功率上限
task_up_limit = sum(task_up); %计算总的可调度数量上限
P_dc_up_limit = sum(P_up);%计算总的可调度功率上限

%% 电动汽车
%500辆车，私家车:公交车:出租车 = 0.8:0.1:0.1
N_ev = 100000;
[P_ev1,P_ev2,P_ev3,~,~,~] = EV(N_ev,[0.9,0.05,0.05],Temperature);
P_ev1 = P_ev1(1:end/T:end);%私家车
P_ev2 = P_ev2(1:end/T:end);%公交车
P_ev3 = P_ev3(1:end/T:end);%出租车
P_ev = P_ev1+P_ev2+P_ev3;

%% 变频空调
start_end = zeros(2,2);
P_ac = zeros(1,86400);
N_ac = 3000;
% T_s = round(unifrnd(22,28,1,1));
% Tin = round(Temperature-unifrnd(3,5,1,1));
T_s = 22;  %23-26温度事宜
Tin = 23.4;

for j = 1:N_ac
    % start_end(1,1) = 12+lognrnd(2.5,0.16,1,1);  %空调开启时间
    % start_end(1,2) = unifrnd(29,33,1,1);
    % start_end(1,1) = normrnd(21,2,1,1);
    % start_end(1,2) = normrnd(31,1.2,1,1);  %空调关闭时间
    start_end(1,1) = round(normrnd(15.1,8.6,1,1),1);
    start_end(1,2) = round(normrnd(21.35,4.29,1,1),1);  %空调关闭时间
    while start_end(1,1) > start_end(1,2) || start_end(1,1) <= 0 || start_end(1,2) > 48
        start_end(1,1) = round(normrnd(15.1,8.6,1,1),1);
        start_end(1,2) = round(normrnd(21.35,4.29,1,1),1);
    end
    
    % while T_s >= Tin
    %     T_s = round(unifrnd(16,28,1,1));
    % end
    [P_AC,T_in,f] = aircondition(T_s,Tin,Temperature,start_end(1,:));
    P_ac = P_ac + P_AC;
end
% for j = 1:1.1*N_ac
%     % start_end(2,2) = unifrnd(16,19,1,1);  %空调关闭时间
%     start_end(2,1) = normrnd(8,0.6,1,1);  %空调开启时间 
%     start_end(2,2) = normrnd(17,2,1,1);  %空调关闭时间
%     while start_end(2,1) > start_end(2,2)
%         start_end(1,1) = normrnd(9,0.6,1,1);
%         start_end(1,2) = normrnd(17,1,1,1);
%     end
%     % while T_s >= Tin
%     %     T_s = round(unifrnd(16,28,1,1));
%     % end
%     [P_AC,T_in,f] = aircondition(T_s,Tin,Temperature,start_end(2,:));
%     P_ac = P_ac + P_AC;
% end
P_ac = P_ac(1:end/T:end);

% figure
% hold on
% plot(P_ac,'linewidth',1.5)

%% 5G基站
Num_5G = 1000;
load('utilization.mat');
x_5g = find(utilization(1:288) == min(utilization(1:288)));
T_5g = utilization(x_5g:x_5g+287)/100;
P_5g = P5G(Num_5G,T_5g);
P_5g = P_5g(1:end/T:end)';
standby_5g = 1.2;
SOC_5g = standby_5g * max(P_5g) * ones(1,T);

%% 机组组合优化初始设置
parpool('local',2);  % 并行优化，设置多核并行

condition = 2;  % 1—数据中心  2—电动汽车   3—变频空调   4—5G基站
load_flexible = [P_dc;P_ev;P_ac;P_5g];
permeability = 0.75; %风光渗透率
flex_percent = 0.5; %灵活性负荷占比
% res = 0.5*ones(1,48); % 响应率（和电价挂钩）
% res = 0.5*ones(3,48);% 电动汽车
res = ones(1,48);

% if condition == 1
% res = readtable('响应度系数', 'Sheet', '数据中心');
% elseif condition == 2
% res = readtable('响应度系数', 'Sheet', '电动汽车');
% elseif condition == 3
% res = readtable('响应度系数', 'Sheet', '变频空调');
% else condition == 4
% res = readtable('响应度系数', 'Sheet', '5G基站');
% end
% res = table2array(res);

times = 1; %循环次数
s = 0.1;  % 极端场景占比
PV1 = zeros(times,T);
LOAD1 = zeros(times,T);
WIND1 = zeros(times,T);

tic
for j = 1:times
% for j = 1:times
% 设置风光和灵活性负荷------------------------
% 取风光负荷数据
if rand(1,1) > s
    P_PV = PV_normal(randi(size(PV_normal,1)),:);
else
    P_PV = PV_abnormal(randi(size(PV_abnormal,1)),:);
end

if rand(1,1) > s
    P_WIND = PWIND_normal(randi(size(PWIND_normal,1)),:);
else
    P_WIND = PWIND_abnormal(randi(size(PWIND_abnormal,1)),:);
end

if rand(1,1) > s
    P_base = LOAD_normal(randi(size(LOAD_normal,1)),:);
else
    P_base = LOAD_abnormal(randi(size(LOAD_abnormal,1)),:);
end

% 选择灵活性负荷类型
P_flexible = load_flexible(condition,:);
% P_flexible = P_dc;  %灵活性负荷(数据中心)
% P_flexible = P_ev;  %灵活性负荷(电动汽车)
% P_flexible = P_ac;  %灵活性负荷(变频空调)
% P_flexible = P_5g;  %灵活性负荷(5G基站)

b(j) =  (flex_percent/(1-flex_percent) * sum(P_base)) / sum(P_flexible);
% b(j) = 1;
P_flexible = b(j) * P_flexible  ; %依照灵活性负荷占比调整灵活性负荷

a(j) = sum(P_PV + P_WIND)/(permeability * sum(P_base + P_flexible));
P_PV = P_PV/a(j);  %依照风光渗透率调整风光出力
P_WIND = P_WIND/a(j);

PV1(j,:) = P_PV;
WIND1(j,:) = P_WIND;
LOAD1(j,:) = P_base;

%定义机组组合变量-------------------------
u = binvar(N, T, 'full'); %定义0-1变量，用于模拟发电机组启停
u0 = ones(1,N);
P_gen = sdpvar(N, T, 'full');   %定义实数变量，发电机组每时刻输出功率
P_wind = sdpvar(1, T, 'full');  %风力发电输出功率
P_pv = sdpvar(1, T, 'full');    %光伏发电输出功率
cost_start = sdpvar(N,T,'full');    %启动成本
cost_stop = sdpvar(N,T,'full');    %关停成本
st = [];%约束条件
z = 0;%目标函数

% 数据中心可转移功率负载
P_trans = sdpvar(4,T,'full');
% 电动汽车转移负载
P_ev_trans = sdpvar(2,T,'full'); 
% 变频空调转移负载
P_ac_trans = sdpvar(1,T,'full'); 
% 5G基站电池容量转移负载
u_ch_5g = binvar(1,T,'full'); % 充电启停
u_dis_5g = binvar(1,T,'full'); % 放电启停
P_ch_5g = sdpvar(1,T,'full');
P_dis_5g = sdpvar(1,T,'full');
n_ch_5g = 0.9;
n_dis_5g = 0.9;
P_5g_trans = sdpvar(1,T,'full'); 

%% 机组约束
%发电机组出力约束
for t = 1:T
    for i = 1:N
        st = [st, u(i,t)* Pmin <= P_gen(i,t) <= u(i,t)* Pmax];
    end
end

% 发电机启停时间约束
for t = 2:T
    for i = 1:N
        range1 = t: min(T,t-1+T_stop(i));
        st = [st, sum(1 - u(i,range1)) >= T_stop(i)*(u(i,t-1) - u(i,t))];  %停机时间约束
        range2 = t: min(T,t-1+T_start(i));
        st = [st, sum(u(i,range2)) >= T_start(i)*(u(i,t) - u(i,t-1))]; %启动时间约束
    end
end

% 发电机爬坡出力约束
for t = 2:T
    for i =1:N
        st = [st, P_gen(i,t) - P_gen(i,t-1) <= u(i,t-1)* R_up(i) + (1-u(i,t-1))* Pmin(i)];
        st = [st, P_gen(i,t-1) - P_gen(i,t) <= u(i,t)* R_down(i) + (1-u(i,t))* Pmin(i)];
    end
end

% 风电光伏出力约束
for t = 1:T
    st = [st, 0 <= P_wind(t) & P_wind(t)<= P_WIND(t)];
    st = [st, 0 <= P_pv(t) & P_pv(t) <= P_PV(t)];
end

%% 数据中心负载转移
if condition == 1
    P_dc_various1 = b(j) * res.* P_dc_various ; %依照灵活性负荷占比调整数据中心参数
    P_up1 = b(j) * res.* P_up ;
    % 可转移负载功率不超过上限
    for t = 1:T
        st = [st, 0 <= P_trans(1,t) & P_trans(1,t) <= P_up1(1,t)];
        st = [st, 0 <= P_trans(2,t) & P_trans(2,t) <= P_up1(2,t)];
        st = [st, 0 <= P_trans(3,t) & P_trans(3,t) <= P_up1(3,t)];
        st = [st, 0 <= P_trans(4,t) & P_trans(4,t) <= P_up1(4,t)];
    end

    %可转移负载功率延迟时间内转移
    for i = 1:4
        for t = 1:T
            st = [st, P_dc_various1(i,t) <= sum(P_trans(i,t:min(t + tt(i),T)))];
            st = [st, P_trans(i,t) - P_dc_various1(i,t) <= sum(P_dc_various1(i,1:t-1) - P_trans(i,1:t-1))];
        end
    end

    %各类负载功率保持不变
    st = [st, sum(P_trans(1,:)) == sum(P_dc_various1(1,:))];
    st = [st, sum(P_trans(2,:)) == sum(P_dc_various1(2,:))];
    st = [st, sum(P_trans(3,:)) == sum(P_dc_various1(3,:))];
    st = [st, sum(P_trans(4,:)) == sum(P_dc_various1(4,:))];

    %各类负载功率不超过原有上限
    st = [st, max(P_trans(1,:)) <= max(P_dc_various1(1,:))];
    st = [st, max(P_trans(2,:)) <= max(P_dc_various1(2,:))];
    st = [st, max(P_trans(3,:)) <= max(P_dc_various1(3,:))];
    st = [st, max(P_trans(4,:)) <= max(P_dc_various1(4,:))];

    %转移后功率
    P_trans_sum = sum(P_trans);
    P_trans_after_dc = P_trans_sum + (1-res.*pp) .* P_dc * b(j);

%% 电动汽车
elseif condition == 2
    for t= 1:T
        st = [st, 0 <= P_ev_trans(1,t)];
        st = [st, 0 <= P_ev_trans(2,t)];
    end
    x_ev3 = 0.5;
    st = [st, sum(P_ev_trans(1,:)) ==sum(b(j) *  res(1,:).*P_ev1)];% 私家车全部可调度
    st = [st, sum(P_ev_trans(2,:)) == sum(b(j) * res(1,:).* P_ev3)];% 出租车部分可调度

    P_ev_after = P_ev_trans(1,:)  + (1-res(1,:)) * b(j) .* P_ev1 + P_ev_trans(2,:) + (1-res(1,:)) * b(j) .* P_ev3 + b(j) * P_ev2;

%% 变频空调
elseif condition == 3
    h_ac = 3; %最大可削减时长/h
    x_ac = res.* (90.5126*exp(-5.5503*h_ac)+45.0894*exp(-0.0028*h_ac))/100; % 最大可削减占比
    % y = [0.25	67.67
    % 0.50	50.57
    % 0.75	46.59
    % 1.00	45.38
    % 1.25	44.98
    % 1.50	44.85
    % 1.75	44.81
    % 2.00	44.79
    % 2.25	44.79
    % 2.50	44.78
    % 2.75	44.78
    % 3.00	44.78];
    for t = 1:T
        st = [st, b(j)*(1-x_ac).*P_ac(1,t) <= P_ac_trans(1,t) <= b(j)*P_ac(1,t)]; % 变频空调部分可削减
    end

    P_ac_after = P_ac_trans;

%% 5G基站
elseif condition == 4
    % P_5g_down = mean(b(j)*P_5g);
    P_5g_trans(1) = b(j)*P_5g(1);
    x_soc_5g =  0.5*Num_5G; % 充放电功率系数
    for t = 1:T
        st = [st, 0 <= u_ch_5g(t)+u_dis_5g(t) <= 1];
        st = [st, 0 <= P_dis_5g(t) <= res(t)*u_dis_5g(t)*x_soc_5g*b(j)];
        st = [st, 0 <= P_ch_5g(t) <= res(t)*u_ch_5g(t)*x_soc_5g*b(j)];
        % st = [st, b(j)*P_5g(t) <= P_5g_trans(t) <= b(j)*SOC_5g(t)];
        % st = [st, b(j)*0.5*(2-res(t))*SOC_5g(t) <= P_5g_trans(t) <= b(j)*0.9*res(t)*SOC_5g(t)];
    end
    for t = 2:T
        st = [st, P_5g_trans(t) == P_5g_trans(t-1) + n_ch_5g * P_ch_5g(t) - n_dis_5g * P_dis_5g(t)];
        st = [st, b(j)*0.5*P_5g(t) <= P_5g_trans(t) <= b(j)*0.9*SOC_5g(t)];
    end

end

%% 负荷功率平衡----------------
if condition == 1
    %%% 数据中心
    for t = 1:T
        st = [st, P_wind(t) + P_pv(t) + sum(P_gen(:,t)) == P_base(t) + P_trans_after_dc(t)];
    end

elseif condition == 2
    %%% 电动汽车
    for t = 1:T
        st = [st, P_wind(t) + P_pv(t) + sum(P_gen(:,t)) == P_base(t) + P_ev_after(t)];
    end

elseif condition == 3
    %%% 变频空调
    for t = 1:T
        st = [st, P_wind(t) + P_pv(t) + sum(P_gen(:,t)) == P_base(t) + P_ac_after(t)];
    end

elseif condition == 4
    %%% 5G基站
    for t = 1:T
        st = [st, P_wind(t) + P_pv(t) + sum(P_gen(:,t)) + P_dis_5g(t) == P_base(t) + P_ch_5g(t)];
    end

end

%% 目标函数
% 启停成本约束---------------
for i = 1:N  % 启停成本条件约束
    for t = 2:T
        cost_start(i,t) = C_start(i)*(u(i,t)-u(i,t-1));
        cost_stop(i,t) = C_stop(i)*(u(i,t-1)-u(i,t));
    end
    cost_start(i,1) = C_start(i)*(u(i,1)-u0(i));% 初始状态下的启停成本
    cost_stop(i,1) = C_stop(i)*(u0(i)-u(i,1));
end

for t = 1:T
    z = z + P_gen(:,t)'* Ai * P_gen(:,t) + Bi * P_gen(:,t) + ...
        sum(cost_start(:,t)) + sum(cost_stop(:,t));
end
% 增加转移惩罚因子减小波动
cc1 = 0.01;
cc2 = 10;
cc3 = 0.01;
cc4 = 0.01;
pc1 = 0;
pc2 = 0;
pc3 = 0;
pc4 = 0;

if condition == 1
    pc1 = cc1 * sum(abs(b(j)*P_dc - P_trans_after_dc));  % 数据中心惩罚因子
elseif condition == 2
    pc2 = cc2 * sum(abs((b(j)*P_ev - P_ev_after)));  % 电动汽车惩罚因子
elseif condition == 3
    pc3 = cc3 * sum(abs((b(j)*P_ac - P_ac_after)));  % 变频空调惩罚因子
elseif condition == 4
    pc3 = cc4 * sum(abs((b(j)*P_5g - P_5g_trans)));  % 变频空调惩罚因子
end

z = z + pc1 + pc2 + pc3 + pc4;%增加惩罚因子

% 设置求解器
ops = sdpsettings('verbose', 2,'solver','cplex');
ops.cplex.dettimelimit = 1000;
ops.cplex.timelimit = 1000;

% 求解
r = solvesdp(st,z,ops);

z1(j) = value(z);
u = value(u);
P_gen1(j,:) = sum(value(P_gen));
% Pgen1(j,:) = P_gen1(1,:); % 发电机1
% Pgen2(j,:) = P_gen1(2,:);
% Pgen3(j,:) = P_gen1(3,:);
P_wind1(j,:) = value(P_wind);
P_pv1(j,:) = value(P_pv);

if condition == 1
    % 数据中心
    P_trans_sum1(j,:) = value(P_trans_sum);
    P_trans_after_dc1(j,:) = value(P_trans_after_dc);
    pc(j) = value(pc1);

elseif condition == 2
    % 电动汽车
    P_ev_after1(j,:) = value(P_ev_after);
    P_trans_ev1(j,:) = value(P_ev_trans(1,:))  + (1-res(1,:)) * b(j) .* P_ev1;
    P_trans_ev3(j,:) = value(P_ev_trans(2,:))  + (1-res(1,:)) * b(j) .* P_ev3;
    pc(j) = value(pc2);

elseif condition == 3
    % 变频空调
    P_ac_after1(j,:) = value(P_ac_after);
    pc(j) = value(pc3);

elseif condition == 4
    % 5G基站
    P_5g_after1(j,:) = value(P_5g_trans);
    u_ch_5g1(j,:) = value(u_ch_5g);
    u_dis_5g1(j,:) = value(u_dis_5g);
    P_ch_5g1(j,:) = value(P_ch_5g);
    P_dis_5g1(j,:) = value(P_dis_5g);
    pc(j) = value(pc4);

end
% figure
% bar(P_ch_5g1);hold on
% bar(-P_dis_5g1)
% % 
% figure
% plot(b(1)*P_5g);hold on
% plot(P_5g_after1)

end
toc

%% 校核优化结果正确性
test_acc = [];
use_index = [];
test_load = [];

if condition == 1
    % 数据中心
    P_after = P_trans_after_dc1;
    P_before = P_dc;

elseif condition == 2
    % 电动汽车
    P_after = P_ev_after1;
    P_before = P_ev;

elseif condition == 3
    % 变频空调
    P_after = P_ac_after1;
    P_before = P_ac;

elseif condition == 4
    % 5G基站
    P_after =  P_ch_5g1-P_dis_5g1;
    P_before = P_5g;
end

for i = 1:times
    if abs(sum(LOAD1(i,:)+P_after(i,:))-...
            sum(P_wind1(i,:)+P_pv1(i,:)+P_gen1(i,:)))<1e-4
        test_acc(i) = 1;
    else
        test_acc(i) = 0;
    end
end
use_index = find(test_acc == 1);
disp(['机组调度成功率：',num2str(100*length(use_index)/times),'%'])

if condition == 4
    % 5G基站
    P_after =  b'*P_5g   + P_ch_5g1-P_dis_5g1;
    P_before = P_5g;
end

for i = 1:length(use_index)
    ii = use_index(i);
    if abs(sum(P_after(ii,:))-...
            sum(b(ii)*P_before)) < 5
        test_load(i) = 1;
    else
        test_load(i) = 0;
    end
end
disp(['负荷调度正确率：',num2str(100*length(find(test_load == 1))/times),'%'])

%% 评估指标体系
ppr = []; 
wpr = [];
E_adjust = [];
up_down_times = zeros(length(use_index),T);
up_down_wd = zeros(length(use_index),T);
E_respondrate = [];
E_respondex = [];

%------------------发电侧指标-----------------
% 风光消纳率 
for j = 1:length(use_index)
    ii = use_index(j);
    ppr(j) = sum(P_pv1(ii,:))/sum(PV1(ii,:));    %光伏消纳率
    wpr(j) = sum(P_wind1(ii,:))/sum(WIND1(ii,:));   %风电消纳率
end

% 负荷调节能力系数
for j = 1:length(use_index)
    ii = use_index(j);
    deta1 = PV1(ii,:) + WIND1(ii,:) - LOAD1(ii,:) - P_after(ii,:);
    deta2 = PV1(ii,:) + WIND1(ii,:) - b(ii)*P_before - LOAD1(ii,:);
    % E_adjust(j) = corr(deta1',deta2','type','Spearman');
    E_adjust(j) = 1-norm([deta1-deta2],1)/sum(abs(deta2));
end

% figure
% plot(deta1,'o-','LineWidth',1); hold on
% plot(deta2,'o-','LineWidth',1);
% legend('实际响应情况','负荷调节期望值')
% 
% figure
% plot(P_trans_after_dc1(6,:)); hold on
% plot(P_dc)


% 负荷响应概率
for j = 1:length(use_index)
    ii = use_index(j);
    deta = b(ii)*P_before - P_after(ii,:);
    dif = P_pv1(ii,:) + P_wind1(ii,:) - b(ii)*P_before - LOAD1(ii,:);
    up_down_times(j,find(deta>0)) = -1;  %负荷实际下降次数
    up_down_times(j,find(deta<0)) = 1;   %负荷实际上升次数
    up_down_wd(j,find(dif>0)) = 1;       %负荷应该上升次数
    up_down_wd(j,find(dif<0)) = -1;      %负荷应该下降次数
    E_respondrate(j) = length(find(up_down_times(j,:) == up_down_wd(j,:)))/T;
end

% 负荷响应期望
e_respondex = zeros(length(use_index),T);
for j = 1:length(use_index)
    ii = use_index(j);
    deta = b(ii)*P_before - P_after(ii,:);
    dif = P_pv1(ii,:) + P_wind1(ii,:) - b(ii)*P_before - LOAD1(ii,:);
    up_down_times(j,find(deta>0)) = -1;  %负荷实际下降次数
    up_down_times(j,find(deta<0)) = 1;   %负荷实际上升次数
    up_down_wd(j,find(dif>0)) = 1;       %负荷应该上升次数
    up_down_wd(j,find(dif<0)) = -1;      %负荷应该下降次数
    e_respondex(j,find(up_down_times(j,:) == up_down_wd(j,:))) = 1;
    e_respondex(j,:) = e_respondex(j,:) .* abs(deta);
end
E_respondex = mean(e_respondex');

% 标准化机组运行成本
E_unitprice = (z1-pc)./ (T*(Pmax' * Ai * Pmax + Bi * Pmax))/flex_percent;

%-------------------用电侧指标-------------------
% 负荷灵活性调节指数
up_max = zeros(1,length(use_index));
down_max = zeros(1,length(use_index));
up_time = zeros(1,length(use_index));
down_time = zeros(1,length(use_index));
up_speed = zeros(1,length(use_index));
down_speed = zeros(1,length(use_index));

for j = 1:length(use_index)
    ii = use_index(j);
    deta = b(ii)*P_before - P_after(ii,:);
    dif = P_pv1(ii,:) + P_wind1(ii,:) - P_before - LOAD1(ii,:);
    up_max(j) = max([abs(deta(find(deta<0))) 0])/mean(LOAD1(ii,:));            % 标准化向上调节最大值
    down_max(j) = max([abs(deta(find(deta>0))) 0])/mean(LOAD1(ii,:));          % 标准化向下调节最大值
    up_time(j) = findLongestConsecutive(find(deta<0))/T;    % 标准化向上调节持续最大时间
    down_time(j) = findLongestConsecutive(find(deta>0))/T;  % 标准化向下调节最大持续时间
    dif = diff(P_after(ii,:))./P_after(ii,1:end-1); 
    % dif = diff(P_after(ii,:))./mean(LOAD1(ii,:))/flex_percent; 
    up_speed(j) = max(abs(dif(find(dif>0))));             % 向上响应最大速率
    down_speed(j) = max(abs(dif(find(dif<0))));           % 向下响应最大速率
    % E_flexindex(j) = length(find(up_down_times(j,:) == up_down_wd(j,:)));
end
up_speed(find(up_speed>100)) = 1;
down_speed(find(down_speed>100)) = 1;
up_speed(find(isinf(up_speed)==1)) = 0;
down_speed(find(isinf(down_speed)==1)) = 0;
if condition == 3
    up_max = zeros(1,length(use_index));
    up_time = zeros(1,length(use_index));
    up_speed = zeros(1,length(use_index));
end
% corr([up_max; down_max; up_time; down_time; up_speed; down_speed]');
% 
% figure
% plot([up_max; down_max; up_time; down_time; up_speed; down_speed]');

%----------------指标体系-----------------
% 计算权重（CRITIC法）
% 标准化
e_use = mapminmax([up_max; down_max; up_time; down_time; up_speed; down_speed],0,1)';
e_gen = mapminmax([ppr; wpr; E_adjust; E_respondrate; E_respondex],0,1)';
e_gen(:,6) = mapminmax(E_unitprice,1,0)';

w_use = CRITIC(e_use);
w_gen = CRITIC(e_gen);

% 熵权法
% data = e_use;
% [m, n] = size(data);
% p = zeros(m,n);
% for j = 1:n
%     p(:,j) = data(:,j) / sum(data(:,j));
% end
% for j = 1:n
%    E(j) = -1 / log(m) * sum(p(:,j) .* log(p(:,j)));
% end
% w_use = (1 - E) / sum(1 - E);

%% 结果可视化
ff = 1;  % 场景ff

% 整体出力情况
figure
area(P_wind1(ff,:)+P_pv1(ff,:)+P_gen1(ff,:),'EdgeColor','none','FaceColor',[82 143 173]/255);hold on
area(P_wind1(ff,:)+P_pv1(ff,:),'EdgeColor','none','FaceColor',[255 230 183]/255);
area(P_wind1(ff,:),'EdgeColor','none','FaceColor',[170 220 224]/255);
area(-LOAD1(ff,:)-P_after(ff,:),'EdgeColor','none','FaceColor',[247 170 88]/255);
area(-P_after(ff,:),'EdgeColor','none','FaceColor',[231 98 84]/255);
plot(WIND1(ff,:),'--','LineWidth',1.5); hold on
plot(PV1(ff,:)+WIND1(ff,:),'--','LineWidth',1.5);
if condition == 1
    legend('火电机组出力','光伏出力','风电出力','基础负荷','数据中心','风电最大出力','光伏最大出力')
elseif condition == 2
    legend('火电机组出力','光伏出力','风电出力','基础负荷','电动汽车','风电最大出力','光伏最大出力')
elseif condition == 3
    legend('火电机组出力','光伏出力','风电出力','基础负荷','变频空调','风电最大出力','光伏最大出力')
else condition == 4
    legend('火电机组出力','光伏出力','风电出力','基础负荷','5G基站','风电最大出力','光伏最大出力')
end
% axis([1 48 -3000 3000])

% for ff =1: 100
% % 负荷响应情况
% figure
% plot(b(ff)*P_before,'linewidth',1.5);hold on
% plot(P_after(ff,:),'linewidth',1.5)
% legend('响应前','响应后')
% axis([1 48 60 150])
% end

% 风光负荷曲线
figure
plot(WIND1(ff,:)); hold on
plot(PV1(ff,:)+WIND1(ff,:));
plot(LOAD1(ff,:))

% 用电侧指标结果
% figure
% plot(wpr,'o-','linewidth',1); hold on
% plot(ppr,'o-','linewidth',1);
% plot(E_respondrate*100,'o-','linewidth',1);
% plot(E_adjust*100,'o-','linewidth',1);


%% 保存结果
% saveresult([num2str(flex_percent),'数据中心'],times,permeability,flex_percent,s,...
%                     w_use,up_max,down_max,up_time,down_time,up_speed,down_speed,...
%                     ppr,wpr,E_respondrate,E_adjust,...
%                     WIND1,PV1,LOAD1,P_trans_after_dc1)

if condition == 1
    path = '结果保存 - 2025-10-2\数据中心_v2';
elseif condition == 2
    path = '结果保存 - 2025-10-2\电动汽车_v2';
elseif condition == 3
    path = '结果保存 - 2025-10-2\变频空调_v2';
else condition == 4
    path = '结果保存 - 2025-10-2\5G基站_v2';
end
% ,char(load_name(condition))
sheetname = ['实验',num2str(times)];
saveresult(path,sheetname,times,permeability,flex_percent,s,...
                    w_use,w_gen,...
                    up_max,down_max,up_time,down_time,up_speed,down_speed,...
                    ppr,wpr,E_adjust,E_respondrate,E_respondex,E_unitprice,...
                    WIND1,PV1,LOAD1,b'.*P_before,P_after)


