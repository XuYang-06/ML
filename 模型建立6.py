import numpy as np
import matplotlib.pyplot as plt
import torch
import math
from torch import nn
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 调用
set_seed(111)

class GasTransportModel:
    def __init__(self, T, p, d, r, L, V_L, P_L, P_g,l,Beta,P_m,P_f,W_F, W_t, W_K,Rho_c,Phi_m,dt):
        # 基本物理常数
        self.k = 1.38e-23         # Boltzmann constant (J/K)
        self.R = 8.314            # Ideal gas constant (J/mol·K)
        self.pi = math.pi
        self.M_CH4 = 16.0425      # CH4 分子量 (g/mol)

        # 初始化输入参数
        self.T = T                # 温度（K）
        self.p = p                # 压力（Pa）
        self.d = d                # 分子直径（m）
        self.r = r                # 孔隙半径（nm）
        self.L = L                # 孔隙特征直径（m）
        self.V_L = V_L            # Langmuir体积（cm3/g）
        self.P_L = P_L            # Langmuir压力（MPa）
        self.P_g = P_g            # 气体压力（MPa）
        self.l = l                # 煤基质裂缝间距(m)
        self.Beta = Beta          # 动扩散系数的衰减系数
        self.P_m = P_m            # 煤基质内的气体压力(Pa)
        self.P_f = P_f            # 煤裂隙的气体压力(Pa)
        self.P_m_array = [P_m]
        self.P_f_array = [P_f]
        self.W_F = W_F            # Fick扩散在煤基质中所占比例(%)
        self.W_t = W_t            # 过渡扩散在煤基质中所占比例(%)
        self.W_K = W_K            # 努森扩散在煤基质中所占比例(%)
        self.Rho_c = Rho_c        # 煤的密度(kg/m3)
        self.Phi_m = Phi_m        # 煤基质的孔隙度(%)
        self.t = 0                # 时间(s)
        self.dt = dt              # 时间步长 (s)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def adsorption_volume(self):
        """吸附气体的含量，cm3/g"""
        return (self.V_L * self.P_g) / (self.P_L + self.P_g)

    def mean_free_path(self):#含变量p
        """分子平均自由程，m"""
        return self.k * self.T / (math.sqrt(2) * self.pi * self.d**2 * self.p)

    def knudsen_number(self):#含变量p
        """Knudsen数，无量纲"""
        return self.mean_free_path() / self.L

    def D_knudsen(self):
        """努森扩散系数，m2/s"""
        return (2 * self.r / 3) * math.sqrt(8 * self.R * self.T / (self.pi * self.M_CH4))

    def Mu_CH4(self,p = None):#含变量p
        """甲烷气体粘度，Pa∙s"""
        if p is not None:
            self.p = p
        return 0.0123 - 1.781e-4 * pow(self.p , 1.216)

    def D_fick(self,p = None):#含变量p
        """为Fick扩散系数，m2/s"""
        return self.k * self.T / (6 * self.pi * self.Mu_CH4(p) * self.r)

    def D_t(self,p = None):#含变量p
        """努森扩散和分子扩散协同扩散系数（过渡扩散系数），m2/s"""
        D_F = self.D_fick(p)
        D_K = self.D_knudsen()
        return (D_F * D_K) / (D_F + D_K)

    def D_e(self,p = None):#含变量p
        """有效扩散系数，m2/s"""
        W_F = self.W_F
        W_t = self.W_t
        W_K = self.W_K
        D_F = self.D_fick(p)
        D_K = self.D_knudsen()
        D_T = self.D_t(p)
        return 1 / (W_F / D_F + W_t / D_T + W_K / D_K)

    def Sigma_C(self):
        """形状因子，m-2"""
        return 3*self.pi / (self.l ** 2)

    def c_m(self):
        """煤基质内的甲烷浓度，kg/m3"""
        return self.M_CH4 * self.P_m / (self.R * self.T)

    def c_f(self):
        """煤裂隙的甲烷浓度，kg/m3"""
        return self.M_CH4 * self.P_f / (self.R * self.T)

    def Q_m(self):
        """单位体积煤基质的气体传输速率,kg/(m3∙s)"""
        D_e = self.D_e()
        Sigma_C = self.Sigma_C()
        c_m = self.c_m()
        c_f = self.c_f()
        return Sigma_C * (c_m - c_f) * D_e * math.exp(-self.Beta * self.t)

    def Z_CH4(self,p = None):
        """50℃条件下甲烷气体的压缩因子"""
        if p is not None:
            self.p = p
        return 4.225e-9 * pow(self.p  ,5) - 6.952e-7 * pow(self.p ,4) + 3.242e-5 * pow(self.p,3) - 9.121e-5 * pow(self.p,2) - 0.0134 * self.p + 0.9994

    def Rho_CH4(self,p = None):
        """甲烷的密度，kg/m3"""
        if p is not None:
            self.p = p
        Z_CH4 = self.Z_CH4(p)
        return self.p * self.M_CH4 / (Z_CH4 * self.T * self.R)

    def m_m(self,p = None):
        """单位煤基质内的气体总质量，kg/m3"""
        if p is not None:
            self.p = p
        m = self.V_L * self.P_m * self.Rho_c * self.Rho_CH4(p) /(self.P_m + self.P_L) + self.Phi_m * self.M_CH4 * self.P_m / (self.R * self.T)
        return m

GAS = GasTransportModel(323,None,1,1,1,13.68,2.45,15,0.005,1e-5,1.5,
                        1.5,37.1,61.3,1.6,1380,5.11,1)

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),nn.ReLU(),
            nn.Linear(64, 64), nn.Tanh(),nn.ReLU(),
            nn.Linear(64, 64), nn.Tanh(),nn.ReLU(),
            nn.Linear(64, 3),nn.ReLU()  # 输出 [P, P_m, K]
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))
def compute_loss(model,GAS,pf_0,t,Beta,M_M,phi_f0,Beta_f,epsilon_L, Beta_m,Pm_0,K0,t_initial,q = 7.72e-6):


    pf_0.requires_grad = True
    t.requires_grad = True
    out = model(pf_0,t)

    Pf = out[:, 0:1]
    P_m = out[:, 1:2]
    K = out[:, 2:3]  # 预测的K值

    # Pf: 0~20 MPa, Pm: 0~20 MPa, K: 正值
    Pf = 20 * torch.sigmoid(out[:, 0:1])
    P_m = 20 * torch.sigmoid(out[:, 1:2])
    K = 1e-15 * torch.exp(out[:, 2:3])  # 渗透率 >0


    Pf_t = torch.autograd.grad(Pf, t, grad_outputs=torch.ones_like(Pf), create_graph=True)[0]
    Pf_x = torch.autograd.grad(Pf, pf_0, grad_outputs=torch.ones_like(Pf), create_graph=True)[0]
    Pf_xx = torch.autograd.grad(Pf_x, pf_0, grad_outputs=torch.ones_like(Pf_x), create_graph=True)[0]
    Pf_tt = torch.autograd.grad(Pf_t, t, grad_outputs=torch.ones_like(Pf_x), create_graph=True)[0]
    P_m_t = torch.autograd.grad(P_m, t, grad_outputs=torch.ones_like(P_m), create_graph=True)[0]

    Sigma_C = GAS.Sigma_C()
    Rho_CH4 = GAS.Rho_CH4(P_m)
    D_e = GAS.D_e(P_m)
    P_L = GAS.P_L
    R = GAS.R
    V_L = GAS.V_L
    T= GAS.T
    Rho_c = GAS.Rho_c
    Phi_m = GAS.Phi_m
    M_CH4 = GAS.M_CH4
    Mu_CH4 = torch.clamp(GAS.Mu_CH4(P_m),min=1e-6,max=10000)
    # print(Mu_CH4)
    # 在 compute_loss 开头统一处理
    Beta = Beta.view(1, 1)
    Beta_f = Beta_f.view(1, 1)
    Beta_m = Beta_m.view(1, 1)
    epsilon_L = epsilon_L.view(1, 1)
    K0 = K0.view(1, 1)
    q = q.view(1, 1)
    Pm_0 = Pm_0.view(1, 1)
    t_initial = t_initial.view(-1, 1)

    #pm 残差
    tem1 = Sigma_C * Rho_c * M_CH4 * D_e * torch.exp(-Beta * t) * (P_m - Pf) * ((P_m + P_L) ** 2)
    tem2 = V_L * R * T * P_L * Rho_c * Rho_CH4 + Phi_m * M_CH4 * ((P_m + P_L) ** 2)

    res_pm = P_m_t - tem1 / tem2

    #q残差(K与/much4没有进行散度处理)
    temp1 = phi_f0 * Pf_t - Sigma_C * D_e * torch.exp(-Beta * t) * (P_m - Pf)
    temp2 = (K / Mu_CH4) * ( (Pf_x - Pf_t) ** 2 + (Pf_tt + Pf_xx) * Pf)
    res_q = q - (temp1 - temp2)


    #K
    b = 0.251 * (K ** 0.36)
    term1 = 1 + b / Pf
    term2 = 1 + (1 / (M_M * phi_f0)) * (Beta_f * (Pf - pf_0) + Beta_m * (P_m - Pm_0))
    term3 = (epsilon_L / phi_f0) * (K / M_M - 1) * (Pf / (P_L + P_m) - Pm_0 / (P_L + Pm_0))
    K_calculated = K0 * term1 * ((term2 + term3) ** 3)
    res_K = K - K_calculated


    #边界
    t_initial.requires_grad = True
    out2 = model(pf_0,t_initial)
    Pf2 = out2[:, 0:1]
    P_m2 = out2[:, 1:2]
    K2 = out2[:, 2:3]
    res_pf_initial = Pf2 - pf_0
    res_P_m_initial = P_m2 - Pm_0

    #使用交叉熵损失函数
    loss_pf_initial = torch.mean( res_pf_initial ** 2)
    loss_P_m_initial = torch.mean( res_P_m_initial ** 2)
    loss_K = torch.mean( res_K ** 2)
    loss_pm = torch.mean( res_pm ** 2)
    loss_q = torch.mean( res_q ** 2)


    #total_loss = res_pf_initial + res_P_m_initial + res_K + res_pm + res_q
    total_loss = loss_pf_initial + loss_P_m_initial + loss_K + loss_pm #+ loss_q
    with torch.no_grad():
        for name, v in [('res_pm', res_pm), ('res_q', res_q), ('res_K', res_K),
                        ('tem2', tem2), ('term2+term3', term2 + term3)]:
            if torch.isnan(v).any():
                pass
                #print('NaN at', name, v)

    return total_loss


def compute_q(model,GAS,pf_0,t,Beta,M_M,phi_f0,phi_f):
    pf_0.requires_grad = True
    t.requires_grad = True
    out = model(pf_0, t)
    Pf = out[:, 0:1]
    P_m = out[:, 1:2]
    K = out[:, 2:3]  # 预测的K值

    Pf_t = torch.autograd.grad(Pf, t, grad_outputs=torch.ones_like(Pf), create_graph=True)[0]
    Pf_x = torch.autograd.grad(Pf, pf_0, grad_outputs=torch.ones_like(Pf), create_graph=True)[0]
    Pf_xx = torch.autograd.grad(Pf_x, pf_0, grad_outputs=torch.ones_like(Pf_x), create_graph=True)[0]

    Sigma_C = GAS.Sigma_C()
    D_e = GAS.D_e(P_m)
    mu_CH4 = GAS.mu_CH4(P_m)

    q = Pf_t - (K / (phi_f * mu_CH4)) * Pf_xx - (Sigma_C * D_e / phi_f) * (P_m - Pf) * torch.exp(-Beta * t)

    return q

# phi_f = torch.tensor(0.015)
#
# Pm_0 = torch.tensor(1.50)
# beta = torch.tensor(1e-5)
# M_M = torch.tensor(4e3)
# beta_f = torch.tensor(0.8)
# beta_m = torch.tensor(0.9)
# epsilon_L = torch.tensor(100)
# K0 = torch.tensor(3.95 * 10e-15)
# t_initial = torch.tensor(0.0)
# q = torch.tensor(7.72e-6)
# 变量初始化部分
device = 'cuda' if torch.cuda.is_available() else 'cpu'

phi_f   = torch.tensor([0.015], device=device)          # [1]
Pm_0    = torch.tensor([1.50],  device=device)          # [1]
beta    = torch.tensor([1e-5], device=device)           # [1]
M_M     = torch.tensor([4e3],   device=device)          # [1]
beta_f  = torch.tensor([0.8],   device=device)          # [1]
beta_m  = torch.tensor([0.9],   device=device)          # [1]
epsilon_L = torch.tensor([100.0], device=device)        # [1]
K0      = torch.tensor([3.95e-15], device=device)       # [1]
t_initial = torch.tensor([0.0], device=device)          # [1]
t_initial = torch.zeros(50,1).to(device)
q       = torch.tensor([7.72e-6], device=device)        # [1]

# 网络和数据也搬到同一设备
model = PINN().to(device)


Pf_0 = torch.rand(1000, 1)  # 空间位置
t = torch.rand(1000, 1)  # 时间



optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
for epoch in range(1500):
    Pf_0 = 0.2 + 14.8 * torch.rand(50, 1, device=device)
    t    = torch.rand(50, 1, device=device)

    loss = compute_loss(
        model=model, GAS=GAS, pf_0=Pf_0, t=t,
        Beta=beta, M_M=M_M,
        phi_f0=phi_f, Beta_f=beta_f, Beta_m=beta_m,
        Pm_0=Pm_0, K0=K0, t_initial=t_initial,
        q=q, epsilon_L=epsilon_L
    )

    print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()