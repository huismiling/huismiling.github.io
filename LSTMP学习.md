
# LSTMP学习记录

## 传统LSTM公式

传统LSTM公式如下，有4个门：输入门、遗忘门、记忆门、输出门。


$ f_t = \sigma_g(W_f x_t + U_f H_{t-1} + b_f) $

$ i_t = \sigma_g(W_i x_t + U_i H_{t-1} + b_i) $

$ o_t = \sigma_g(W_o x_t + U_o H_{t-1} + b_o) $

$ \tilde{c}_t = \sigma_g(W_c x_t + U_c H_{t-1} + b_c) $

$ c_t = f_t \circ c_{t-1} + i_t \circ \tilde{c}_t  $

$ h_t = o_t \circ \sigma_h(c_t) $

其中：

$ x_{t}\in {R}^{d} $ : input vector to the LSTM unit

$ f_{t}\in {(0,1)}^{h}$ : forget gate's activation vector

$ i_{t}\in {(0,1)}^{h}$ : input/update gate's activation vector

$ o_{t}\in {(0,1)}^{h}$ : output gate's activation vector

$ h_{t}\in {(-1,1)}^{h}$ : hidden state vector also known as output vector of the LSTM unit

$ {\tilde {c}}_{t}\in {(-1,1)}^{h}$ : cell input activation vector

$ c_{t}\in \mathbb {R} ^{h}$ : cell state vector

$ W\in \mathbb {R} ^{h\times d}$ , $ U\in \mathbb {R} ^{h\times h}$ and $ b\in \mathbb {R} ^{h}$ : weight matrices and bias vector parameters which need to be learned during training


## LSTM Projection with PeepHole公式


$ f_{t} = \sigma _{g}(W_{f}*x_{t}+U_{f}*r_{t-1}+V_{f}\circ c_{t-1}+b_{f})$ 

$ i_{t} = \sigma _{g}(W_{i}*x_{t}+U_{i}*r_{t-1}+V_{i}\circ c_{t-1}+b_{i})$ 

$ c_{t}= f_{t}\circ c_{t-1}+i_{t}\circ \sigma _{c}(W_{c}*x_{t}+U_{c}*r_{t-1}+b_{c})$ 

$ o_{t} = \sigma _{g}(W_{o}*x_{t}+U_{o}*r_{t-1}+V_{o}\circ c_{t}+b_{o})$ 

$ h_{t} = o_{t}\circ \sigma _{h}(c_{t})$ 

$ r_{t} = W_{r} h_{t}$ 


