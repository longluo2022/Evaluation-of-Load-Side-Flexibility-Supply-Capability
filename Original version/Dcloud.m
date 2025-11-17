% 定义 Excel 文件名
filename ={ '存档结果\数据中心_v2.xls';
    '存档结果\电动汽车_v2.xls';
    '存档结果\变频空调_v2.xls';
    '存档结果\5G基站_v2.xls' };


% 指定要从中读取数据的工作表
sheet_flex = {'占比0.1','占比0.2','占比0.3','占比0.4','渗透率0.75'}; 
sheet_per = {'渗透率0.15','渗透率0.3','渗透率0.45','渗透率0.75 (2)','渗透率0.75'};
sheet_time = {'峰时0.75','平时0.75','谷时0.75'};
sheet_res = {'渗透率0.75','动态响应度0.5','响应度0.5'};

sheet_name = sheet_flex;  % 选择对应评估场景
condition = 2;
% 指定要从每个工作表中提取的行索引
rowsToExtract = [1:6;9:14]; % 修改为你想提取的行索引
fullData = [];
% 初始化一个 cell 数组来存储提取的数据
dif_flex_use = cell(length(sheet_name), 1);
dif_flex_gen = cell(length(sheet_name), 1);

% 循环遍历每个工作表
for i = 1:length(sheet_name)
    % 读取当前工作表的全部数据到表格变量中
    fullData = readtable(filename{condition}, 'Sheet', sheet_name{i});
for col = 1:width(fullData)
    % 检查列是否为数值型
    if isnumeric(fullData.(col))
        % 将空值替换为零
        fullData.(col)(isnan(fullData.(col))) = 0;
    end
end
    % 提取特定的行
    extractedRows = fullData(rowsToExtract(1,:), :);
    % 将提取的行存储到 cell 数组中
    dif_flex_use{i} = extractedRows;

    extractedRows = fullData(rowsToExtract(2,:), :);
    dif_flex_gen{i} = extractedRows;
end


% 将不同场景同一指标归类
n = 6;
use_index = cell(n,1);
gen_index = cell(n,1);

for i = 1:n
    for j = 1:length(sheet_name)
       use_index{i}(j,:) =  dif_flex_use{j}(i,:);
       gen_index{i}(j,:) =  dif_flex_gen{j}(i,:);
    end
end

% 指标归一化
use_index_normal = cell(n,1);
gen_index_normal = cell(n,1);
for i = 1:n
    index = table2array(use_index{i});
    index = reshape(table2array(use_index{i}),[1,length(sheet_name)*100]);
    use_index_normal{i} = reshape(mapminmax(index,0,1),[length(sheet_name),100]);
    index = table2array(gen_index{i});
    index = reshape(table2array(gen_index{i}),[1,length(sheet_name)*100]);
    gen_index_normal{i} = reshape(mapminmax(index,0,1),[length(sheet_name),100]);
end
index = table2array(gen_index{6});
index = reshape(table2array(gen_index{6}),[1,length(sheet_name)*100]);
gen_index_normal{6} = reshape(mapminmax(index,1,0),[length(sheet_name),100]);
% w = CRITIC(index)
% 将指标分配回不同场景下
dif_flex_use_normal = cell(length(sheet_name),1);
dif_flex_gen_normal = cell(length(sheet_name),1);
for i = 1:n
    for j = 1:length(sheet_name)
        dif_flex_use_normal{j}(i,:) = use_index_normal{i}(j,:);
        dif_flex_gen_normal{j}(i,:) = gen_index_normal{i}(j,:);
    end
end


%% 二维云模型
cloud_num = 3000; % 云滴数

% 权重  
w_use = [0.260786394	0.071363741	0.170520035	0.225590253	0.108528966	0.163210611
0.254467512	0.154046913	0.135025292	0.160753251	0.133143123	0.16256391
0.184824386	0.206396768	0.162186006	0.116505983	0.148595583	0.181491274
0.142817784	0.13569617	0.178679483	0.219589824	0.140261128	0.182955611
0.178216714	0.154063671	0.169357246	0.146714101	0.175468064	0.176180203];

w_gen = [0.180957725	1.05541E-16	0.150144287	0.188455794	0.167088007	0.313354187
0.093925324	0.10839097	0.212662049	0.14618424	0.166095298	0.272742119
0.134889009	0.147544484	0.216905834	0.158818226	0.135851915	0.205990532
0.185223145	0.162341485	0.162907297	0.16474011	0.175119703	0.149668259
0.155208427	0.152058766	0.157208524	0.180282321	0.198298289	0.156943673];

Ex = [];Ey = []; Enx = []; Eny = []; Hex = []; Hey = [];
for j = 1:length(sheet_name)
    for i = 1:n
        [~,~,Ex(j,i),Enx(j,i),Hex(j,i)] = cloudgenerator(n * dif_flex_use_normal{j}(i,:)',cloud_num);
        [~,~,Ey(j,i),Eny(j,i),Hey(j,i)] = cloudgenerator(n * dif_flex_gen_normal{j}(i,:)',cloud_num);
    end
end

Ex1 = []; Ex2 = []; En1 = []; En2 = []; He1 = []; He2 = [];
for j =1:length(sheet_name)
    Ex1(j) = sum(w_use(5,:).*Ex(j,:),2);
    Ex2(j) = sum(w_gen(5,:).*Ey(j,:),2);
    En1(j) = sum(w_use(5,:).*Enx(j,:),2);
    En2(j) = sum(w_gen(5,:).*Eny(j,:),2);
    He1(j) = abs(sum(w_use(5,:).*Hex(j,:),2));
    He2(j) = abs(sum(w_gen(5,:).*Hey(j,:),2));
end
% Ex1 = sum(w_use.*Ex,2);
% Ex2 = sum(w_gen.*Ey,2);
% En1 = sum(w_use.*Enx,2);
% En2 = sum(w_gen.*Eny,2);
% He1 = abs(sum(w_use.*Hex,2));
% He2 = abs(sum(w_gen.*Hex,2));
if length(find(He1 > 1)) || length(find(He2 > 1)) ~= 0
    He1 = He1/4;
    He2 = He2/4;
end
if length(find(En1 > 1)) || length(find(En2 > 1)) ~= 0
    En1 = En1/4;
    En2 = En2/4;
end

% --------------全局归一化--------------------
candicate = [0.891047457	2.8636	1.48726926	0.727352039;
3.086152477	3.9889	3.409339763	3.091028365;
0.07465686	0.2164	0.055285918	0.060506712;
0.213759138	0.3067	0.279030204	0.184181571;
0.067462975	0.2949	0.123579554	0.064548663;
0.134039638	0.3028	0.313551604	0.127912641];

Ex1 = Ex1*candicate(1,condition)/Ex1(1);
Ex2 = Ex2*candicate(2,condition)/Ex2(1);
En1 = En1*candicate(3,condition)/En1(1);
En2 = En2*candicate(4,condition)/En2(1);
He1 = He1*candicate(5,condition)/He1(1);
He2 = He2*candicate(6,condition)/He2(1);

% 不同时段
if strcmp(sheet_name{1}, sheet_time{1}) 
    load_flex = [Ex1;Ex2;En1;En2;He1;He2];
    load_flex(:,4) = candicate(:,condition);
    for i =1 :6
        zz(i) = load_flex(i,4)/(load_flex(i,1)*6/24+load_flex(i,2)*10/24+load_flex(i,3)*8/24);
    end
    load_flex(:,1:3) = zz'.*load_flex(:,1:3);
    load_flex  =load_flex(:,1:3);

    Ex1 = load_flex(1,:);
    Ex2 = load_flex(2,:);
    En1 = load_flex(3,:);
    En2 = load_flex(4,:);
    He1 = load_flex(5,:);
    He2 = load_flex(6,:);
end
% ---------绘制二维云图-----------
% 不同场景颜色
color_cloud = [215 227 191; 189 208 145; 128 156 67; 87 107 46; 39 48 20]/255; % 不同渗透率
% color_cloud = [56 44 71; 109 86 137; 204 194 217]/255;  % 不同时段
% color_cloud = [129 50 44; 192 79 69; 229 185 181]/255;  % 不同响应度
% color_cloud = [252 211 177; 245 170 107; 242 133 43; 201 98 14; 140 68 9]/255; % 不同占比
figure
for j = 1:length(sheet_name)
    x = [];
    y = [];
    Enn1 = [];
    Enn2 = [];
    for i = 1:cloud_num
        % 生成以En为期望 以He^2为方差的正态随机数Enn
        % randn(m)生成m行m列的标准正态分布的随机数或矩阵的函数
        Enn1 = En1(j) + randn(1).* He1(j);
        Enn2 = En2(j) + randn(1).* He2(j);
        % 生成以Ex为期望，以Enn^2为方差的正态随机数x
        x(1,i) = Ex1(j) + randn(1) * abs(Enn1);
        x(2,i) = Ex2(j) + randn(1) * abs(Enn2);

        % 计算隶属度（确定度）
        y(i) = exp(-((x(1,i)-Ex1(j)).^2/(2 * Enn1.^2)+(x(2,i)-Ex2(j)).^2/(2 * Enn2.^2)));
    end 
    plot3(x(1,:), x(2,:), y(1,:), '.','color',color_cloud(j,:));hold on
end
% 绘制二维云图

grid on
xlabel('用电侧','FontName', 'Arial')
ylabel('发电侧','FontName', 'Arial')
set(gca, 'FontName', 'Times New Roman');
axis([0 6 0 6 0 1])

% 绘制标准云
Ex_nn = [5.4, 4.2, 3, 1.8, 0.6];
colors = [20 54 95; 32 87 155; 85 147 221; 137 180 231; 223 235 249]/255;
% Ex_nn = 3;
for j = 1:length(Ex_nn)
    Ex1_satnd = Ex_nn(j);
    Ex2_satnd = Ex_nn(j);
    En1_stand = 0.33;
    En2_stand = 0.33;
    He1_stand = 0.1;
    He2_stand = 0.1;
    for i = 1:cloud_num
        % 生成以En为期望 以He^2为方差的正态随机数Enn
        % randn(m)生成m行m列的标准正态分布的随机数或矩阵的函数
        Enn1 = En1_stand + randn(1).* He1_stand;
        Enn2 = En2_stand + randn(1).* He2_stand;
        % 生成以Ex为期望，以Enn^2为方差的正态随机数x
        x(1,i) = Ex1_satnd + randn(1) * abs(Enn1);
        x(2,i) = Ex2_satnd + randn(1) * abs(Enn2);
        % 计算隶属度（确定度）
        y(i) = exp(-((x(1,i)-Ex1_satnd).^2/(2 * Enn1.^2)+(x(2,i)-Ex2_satnd).^2/(2 * Enn2.^2)));
    end
    plot3(x(1,:), x(2,:), y(1,:), '.','color',colors(5,:));hold on
end


%%  计算云贴合度
% for i = 1:length(Ex_nn)
%     R(i) = 1/sqrt((Ex1-Ex_nn(i))^2+(Ex2-Ex_nn(i))^2);
% end
% disp(['灵活性等级:',num2str(find(R==max(R)))])

ss = zeros(5,length(sheet_name));
s1 = [];
for m =1:20
    for i = 1: 5
        for j =1 : length(sheet_name)
            s1(i,j) = calculateMembership([Ex1(j),En1(j),He1(j)],[Ex2(j),En2(j),He2(j)],[Ex_nn(i),0.33,0.1]);
        end
    end
    ss = ss+s1;
end

ss = ss/20;
class = [];
for j = 1:length(sheet_name)
    class(j) = find(ss(:,j)==max(ss(:,j)));
end
disp(class)
%%% class： 1-极强，2-强，3-中等，4-弱，5-极弱

%% 指标值计算
use_normal = [];
gen_normal = [];
use_cal = candicate(1,2)/candicate(1,condition);
gen_cal = candicate(2,2)/candicate(2,condition);
for i = 1:length(sheet_name)
    use_normal(i,:) = mean(dif_flex_use_normal{i,1}')/use_cal;
    gen_normal(i,:) = mean(dif_flex_gen_normal{i,1}')/gen_cal;
end

