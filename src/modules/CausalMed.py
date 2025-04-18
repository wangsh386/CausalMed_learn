# 导入必要的库
import numpy as np
import torch
import torch.nn as nn

# 从自定义模块导入异构图和同构图处理器
from .hetero_effect_graph import hetero_effect_graph
from .homo_relation_graph import homo_relation_graph

class CausalMed(torch.nn.Module):
    def __init__(
            self,
            causal_graph,        # 预定义的因果图结构，包含实体间因果关系
            tensor_ddi_adj,      # 药物-药物相互作用矩阵，用于正则化约束
            emb_dim,             # 嵌入维度（通常选择128或256）
            voc_size,            # 各实体的词汇表大小 [疾病数, 操作数, 药物数]
            dropout,             # 随机失活率（0.0-1.0）
            device=torch.device('cpu'),  # 计算设备（默认CPU）
    ):
        """基于因果图的药物推荐模型初始化"""
        super(CausalMed, self).__init__()
        self.device = device     # 存储计算设备信息
        self.emb_dim = emb_dim   # 存储嵌入维度

        # 实体嵌入层（疾病、操作、药物）
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(voc_size[0], emb_dim),  # 疾病嵌入层：voc_size[0]疾病数 → emb_dim
            torch.nn.Embedding(voc_size[1], emb_dim),  # 操作嵌入层
            torch.nn.Embedding(voc_size[2], emb_dim)   # 药物嵌入层（注意：需确保voc_size[2] > 5000）
        ])

        # Dropout层配置（仅在有效dropout值时启用）
        if 0 < dropout < 1:
            self.rnn_dropout = torch.nn.Dropout(p=dropout)  # 随机失活层
        else:  # 无效配置时使用空序列
            self.rnn_dropout = torch.nn.Sequential()

        self.causal_graph = causal_graph  # 存储因果图结构

        # 同构图处理器（分别处理疾病、操作、药物的同类关系）
        self.homo_graph = nn.ModuleList([
            homo_relation_graph(emb_dim, device),  # 疾病同构图处理器
            homo_relation_graph(emb_dim, device),  # 操作同构图处理器
            homo_relation_graph(emb_dim, device)   # 药物同构图处理器
        ])

        # 异构图处理器（处理跨类型实体关系，如疾病-药物、操作-药物）
        self.hetero_graph = hetero_effect_graph(emb_dim, emb_dim, device)

        # 特征融合权重参数（可学习参数，形状3x2）
        # rho[0]对应疾病，rho[1]对应操作，rho[2]对应药物
        # 每行两个元素分别表示同构和异构特征的融合权重
        self.rho = nn.Parameter(torch.ones(3, 2))

        # 序列编码器（GRU网络，分别处理不同实体类型的时序信息）
        self.seq_encoders = torch.nn.ModuleList([
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True),  # 疾病序列编码器
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True),  # 操作序列编码器
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True)   # 药物序列编码器
        ])

        # 最终查询层：将患者表征转换为药物推荐分数
        self.query = torch.nn.Sequential(
            torch.nn.ReLU(),  # 激活函数增加非线性
            # 输入维度：GRU隐藏层（emb_dim*3） + 最后时刻输出（emb_dim*3）= 6*emb_dim
            torch.nn.Linear(emb_dim * 6, voc_size[2])  # 输出维度：药物词汇表大小
        )

        self.tensor_ddi_adj = tensor_ddi_adj  # 存储药物相互作用矩阵
        self.init_weights()  # 执行权重初始化

    def init_weights(self):
        """初始化嵌入层权重（均匀分布）"""
        initrange = 0.1  # 初始化范围
        for item in self.embeddings:
            # 对每个嵌入层的权重进行均匀分布初始化
            item.weight.data.uniform_(-initrange, initrange)

    def forward(self, patient_data):
        """
        前向传播过程
        Args:
            patient_data: 患者就诊数据列表，每个元素为单次就诊记录：
                [疾病索引列表, 操作索引列表, 药物索引列表, 时间戳]
        Returns:
            score: 药物推荐分数张量（shape: [1, 药物数]）
            batch_neg: 药物相互作用正则化损失值
        """
        # 初始化序列存储容器（疾病、操作、药物）
        seq_diag, seq_proc, seq_med = [], [], []
        
        # 遍历每个就诊记录（按时间顺序处理）
        for adm_id, adm in enumerate(patient_data):
            # --------------------- 数据预处理 ---------------------
            # 获取当前就诊的疾病和操作索引
            idx_diag = torch.LongTensor(adm[0]).to(self.device)  # 疾病索引转张量
            idx_proc = torch.LongTensor(adm[1]).to(self.device)  # 操作索引转张量
            
            # 获取疾病和操作的初始嵌入（添加dropout）
            # 形状说明：
            # emb_diag: (1, 当前就诊疾病数, emb_dim)
            # emb_proc: (1, 当前就诊操作数, emb_dim)
            emb_diag = self.rnn_dropout(self.embeddings[0](idx_diag)).unsqueeze(0)
            emb_proc = self.rnn_dropout(self.embeddings[1](idx_proc)).unsqueeze(0)

            # ----------------- 同构图关系学习 -----------------
            # 获取因果图中的同构图结构（基于时间戳和实体类型）
            graph_diag = self.causal_graph.get_graph(adm[3], "Diag")  # 疾病图结构
            graph_proc = self.causal_graph.get_graph(adm[3], "Proc")  # 操作图结构
            
            # 通过同构图处理器增强表示
            # 输出形状与输入相同：(1, 实体数, emb_dim)
            emb_diag1 = self.homo_graph[0](graph_diag, emb_diag)  # 疾病同构表示
            emb_proc1 = self.homo_graph[1](graph_proc, emb_proc)  # 操作同构表示

            # 处理药物历史信息（当前就诊的先前药物）
            if adm == patient_data[0]:  # 第一个就诊无历史药物
                # 使用零填充（形状：(1, 1, emb_dim)）
                emb_med1 = torch.zeros(1, 1, self.emb_dim).to(self.device)
            else:  # 非首诊时获取前次就诊药物
                adm_last = patient_data[adm_id - 1]
                idx_med = torch.LongTensor([adm_last[2]]).to(self.device)  # 前次药物索引
                emb_med = self.rnn_dropout(self.embeddings[2](idx_med))  # 药物嵌入
                med_graph = self.causal_graph.get_graph(adm_last[3], "Med")  # 药物图结构
                emb_med1 = self.homo_graph[2](med_graph, emb_med)  # 药物同构表示

            # ---------------- 异构图关系学习 ----------------
            # 处理药物索引（特殊占位符处理）
            if adm == patient_data[0]:  # 首诊处理
                # 使用占位符5000（需确保药物嵌入层包含该索引）
                idx_med = torch.LongTensor([5000]).to(self.device)
                # 对应嵌入初始化为零向量（形状：(1, 1, emb_dim)）
                emb_med = torch.zeros(1, 1, self.emb_dim).to(self.device)
            else:  # 非首诊时获取前次药物
                adm_last = patient_data[adm_id - 1]
                idx_med = torch.LongTensor(adm_last[2]).to(self.device)
                emb_med = self.rnn_dropout(self.embeddings[2](idx_med)).unsqueeze(0)

            # 初始化因果权重矩阵（疾病-药物、操作-药物）
            diag_med_weights = np.zeros((len(idx_diag), len(idx_med)))  # 形状：(疾病数, 药物数)
            proc_med_weights = np.zeros((len(idx_proc), len(idx_med)))  # 形状：(操作数, 药物数)

            # 填充实际因果权重（排除占位符情况）
            if 5000 not in idx_med:  # 有效药物存在时
                # 填充疾病-药物权重
                for i, med in enumerate(idx_med):
                    for j, diag in enumerate(idx_diag):
                        # 从因果图获取疾病对药物的影响权重
                        diag_med_weights[j, i] = self.causal_graph.get_effect(
                            diag.item(), med.item(), "Diag", "Med"
                        )
                
                # 填充操作-药物权重
                for i, med in enumerate(idx_med):
                    for j, proc in enumerate(idx_proc):
                        proc_med_weights[j, i] = self.causal_graph.get_effect(
                            proc.item(), med.item(), "Proc", "Med"
                        )

            # 通过异构图处理器获取跨实体关系表示
            # 输入形状：
            # emb_diag: (1, 疾病数, emb_dim)
            # emb_proc: (1, 操作数, emb_dim)
            # emb_med: (1, 药物数, emb_dim)
            # 输出形状与输入相同
            emb_diag2, emb_proc2, emb_med2 = self.hetero_graph(
                emb_diag, emb_proc, emb_med, 
                diag_med_weights, proc_med_weights
            )

            # ----------------- 特征融合 -----------------
            # 加权融合同构和异构表示（可学习权重rho）
            # 疾病融合：rho[0,0]*同构 + rho[0,1]*异构
            emb_diag3 = self.rho[0, 0] * emb_diag1 + self.rho[0, 1] * emb_diag2
            # 操作融合：rho[1,0]*同构 + rho[1,1]*异构
            emb_proc3 = self.rho[1, 0] * emb_proc1 + self.rho[1, 1] * emb_proc2
            # 药物融合：rho[2,0]*同构 + rho[2,1]*异构
            emb_med3 = self.rho[2, 0] * emb_med1 + self.rho[2, 1] * emb_med2

            # 聚合当前就诊的特征（沿实体维度求和）
            # 输出形状：seq_diag元素为(1, 1, emb_dim)
            seq_diag.append(torch.sum(emb_diag3, keepdim=True, dim=1))
            seq_proc.append(torch.sum(emb_proc3, keepdim=True, dim=1))
            seq_med.append(torch.sum(emb_med3, keepdim=True, dim=1))

        # ----------------- 序列编码 -----------------
        # 拼接所有就诊的序列（沿时间维度）
        # 输出形状：
        # seq_diag: (1, 就诊次数, emb_dim)
        # seq_proc: 同上
        # seq_med: 同上
        seq_diag = torch.cat(seq_diag, dim=1)
        seq_proc = torch.cat(seq_proc, dim=1)
        seq_med = torch.cat(seq_med, dim=1)

        # 分别对疾病、操作、药物序列进行GRU编码
        # 输出说明：
        # output_*: 各时间步的隐藏状态（形状同输入）
        # hidden_*: 最终时刻的隐藏状态（形状: (1, 1, emb_dim)）
        output_diag, hidden_diag = self.seq_encoders[0](seq_diag)
        output_proc, hidden_proc = self.seq_encoders[1](seq_proc)
        output_med, hidden_med = self.seq_encoders[2](seq_med)

        # 拼接GRU的最终隐藏状态（整体序列信息）
        # seq_repr形状: (1, 1, emb_dim*3)
        seq_repr = torch.cat([hidden_diag, hidden_proc, hidden_med], dim=-1)
        
        # 拼接最后时间步的输出（近期状态信息）
        # last_repr形状: (1, emb_dim*3)
        last_repr = torch.cat([
            output_diag[:, -1],  # 最后时刻疾病序列输出
            output_proc[:, -1],  # 操作序列输出
            output_med[:, -1]    # 药物序列输出
        ], dim=-1)
        
        # 生成最终患者表征（展平拼接）
        # patient_repr形状: (emb_dim*6)
        patient_repr = torch.cat([
            seq_repr.flatten(),  # 展平后的隐藏状态（emb_dim*3）
            last_repr.flatten()  # 展平后的最后输出（emb_dim*3）
        ])

        # ----------------- 药物预测 -----------------
        # 生成药物推荐分数
        # score形状: (1, 药物数)
        score = self.query(patient_repr).unsqueeze(0)
        
        # 计算药物相互作用正则化损失
        neg_pred_prob = torch.sigmoid(score)  # 概率化
        # 计算负样本交互矩阵
        neg_pred_prob = torch.matmul(neg_pred_prob.t(), neg_pred_prob)
        # 与DDI矩阵相乘并求和（惩罚高风险组合）
        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        
        return score, batch_neg
