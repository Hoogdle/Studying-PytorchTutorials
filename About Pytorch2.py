## 텐서의 연산
# 텐서에 대한 수학 연산, 삼각함수, 비트 연산, 비교 연산, 집계 등 제공

import torch 
import math

# 차원의 차이
# a = torch.rand(1,2)
# b = torch.rand(2)
# print (a)
# print(b)
# print(a.ndim) #2
# print(b.ndim) #1

# 다양한 연산들
# a = torch.rand(1,2) *2 -1
# print(a) #tensor([[-0.0734,  0.6768]])
# print(torch.abs(a)) #tensor([[0.0734, 0.6768]])
# print(torch.ceil(a)) #tensor([[-0., 1.]])
# print(torch.floor(a)) #tensor([[-1.,  0.]])
# print(torch.clamp(a,-0.5,0.5)) #tensor([[-0.0734,  0.5000]])
# clamp => 최소값 : -0.5 / 최대값 : 0.5 로 제한

# print(a) #tensor([[-0.5816,  0.4900]])
# print(torch.min(a)) #tensor(-0.5816)
# print(torch.max(a)) #tensor(0.4900)
# print(torch.mean(a)) #tensor(-0.0458) #평균
# print(torch.std(a)) #tensor(0.7577) #표준편차
# print(torch.prod(a)) #tensor(-0.2850) #product계산
# print(torch.unique(torch.tensor([1,2,3,1,2,2]))) #tensor([1, 2, 3]) #중복제거

# max,min에 dim 인자를 줄 경우, argmax, argmin 함께 리턴
# argmax == 최대값을 가진 인덱스
# argmin == 최소값을 가진 인덱스
# x = torch.rand(2,2)
# print(x) #tensor([[0.4402, 0.7486],
        # [0.2746, 0.0864]])
# print(x.max(dim=0)) #열을 따라
# torch.return_types.max(
# values=tensor([0.4402, 0.7486]),
# indices=tensor([0, 0]))
# print(x.max(dim=1)) #행을 따라
# torch.return_types.max(
# values=tensor([0.7486, 0.2746]),
# indices=tensor([1, 0]))

# torch.add : 덧셈
x = torch.rand(2,2)
# print(x) #tensor([[0.2198, 0.9375],
#         [0.4319, 0.2805]])
y = torch.rand(2,2)
# print(y) #tensor([[0.5955, 0.0213],
#         [0.2286, 0.6038]])
# 동일결과
# print(x+y) #tensor([[0.8154, 0.9589],
#             [0.6605, 0.8843]])
# print(torch.add(x,y)) # tensor([[0.8154, 0.9589],
#                         [0.6605, 0.8843]])

# 결과 텐서를 인자로 제공
# result = torch.empty(2,2)
# torch.add(x,y, out = result)
# print(result) #tensor([[1.1352, 1.3186],
        #      [0.6799, 0.4579]])

# in-place 방식
# in-place 방식으로 텐서의 값을 변경하는 연산 뒤에는 '_'가 붙음
# print(x) #tensor([[0.5601, 0.6404],
        #[0.2568, 0.9139]])
# print(y) #tensor([[0.6800, 0.1765],
        #[0.5104, 0.4440]])
# y.add_(x)
# print(y) #tensor([[1.2401, 0.8169],
           # [0.7672, 1.3579]])

# tensor.sub : 뺄셈
# print(x)
# tensor([[0.5106, 0.8582],
#         [0.4192, 0.0730]])
# print(y)
# tensor([[0.0799, 0.0667],
#         [0.4467, 0.4835]])
# print(x-y)
# tensor([[ 0.4307,  0.7915],
#         [-0.0275, -0.4105]])
# print(torch.sub(x,y))
# tensor([[ 0.4307,  0.7915],
#         [-0.0275, -0.4105]])
# print(x.sub(y))
# tensor([[ 0.4307,  0.7915],
#         [-0.0275, -0.4105]])

# tensor.mul : 곱셈
# print(x)
# tensor([[0.2794, 0.2937],
#         [0.6867, 0.3301]])
# print(y)
# tensor([[0.9713, 0.1248],
#         [0.0238, 0.3412]])
# print(x*y)
# tensor([[0.2714, 0.0366],
#         [0.0164, 0.1126]])
# print(torch.mul(x,y))
# tensor([[0.2714, 0.0366],
#         [0.0164, 0.1126]])
# print(x.mul(y))
# tensor([[0.2714, 0.0366],
#         [0.0164, 0.1126]])

# tensor.div : 나눗셈
# print(x)
# tensor([[0.4170, 0.0095],
#         [0.1411, 0.1569]])
# print(y)
# tensor([[0.8537, 0.0952],
#         [0.8705, 0.8398]])
# print(x/y)
# tensor([[0.4885, 0.0995],
#         [0.1621, 0.1868]])
# print(torch.div(x,y))
# tensor([[0.4885, 0.0995],
#         [0.1621, 0.1868]])
# print(x.div(y))
# tensor([[0.4885, 0.0995],
#         [0.1621, 0.1868]])

# tensor.mm,torch matmul : 내적(dot product)
# print(x)
# tensor([[0.3870, 0.6274],
#         [0.5625, 0.2947]])
# print(y)
# tensor([[0.8231, 0.6415],
#         [0.6273, 0.6281]])
# print(torch.mm(x,y))
# tensor([[0.7121, 0.6423],
#         [0.6479, 0.5459]])
# z = torch.mm(x,y)
# print(torch.svd(z))
# torch.return_types.svd(
# U=tensor([[-0.7494, -0.6621],
#         [-0.6621,  0.7494]]),
# S=tensor([1.2794, 0.0214]),
# V=tensor([[-0.7524,  0.6588],
#         [-0.6588, -0.7524]]))

# 텐서의 조작(Manipulations)

# 인덱싱(Indexing) : Numpy처럼 인덱싱 형태로 사용가능
# x = torch.Tensor([[1,2],[3,4]])
# print(x)
# tensor([[1., 2.],
#         [3., 4.]])
# print(x[0,0]) tensor(1.)
# print(x[0,1]) tensor(2.)
# print(x[1,0]) tensor(3.)
# print(x[1,1]) tensor(4.)

# print(x[:,0]) tensor([1., 3.])
# print(x[:,1]) tensor([2., 4.])

# print(x[0,:]) tensor([1., 2.])
# print(x[1,:]) tensor([3., 4.])

# view : 텐서의 크기(size) or 모양(shape) 변경
# 기본적으로 변경 전과 후에 텐서 안의 원소 개수가 유지되어야 함
# -1 로 설정되면 계산을 통해 해당 크기값을 유추
# x = torch.randn(4,5)
# print(x)
# tensor([[ 0.6721, -0.7483,  0.7681,  0.2374, -0.6830],
#         [ 1.5633,  1.2590, -0.7781, -0.2972, -0.3517],
#         [-0.4653, -0.0958,  0.6321, -0.0694,  0.2954],
#         [-0.0730,  1.3622,  0.6438,  1.5559,  0.8432]])
# y = x.view(20)
# print(y)
# tensor([ 0.6721, -0.7483,  0.7681,  0.2374, -0.6830,  1.5633,  1.2590, -0.7781,        
#         -0.2972, -0.3517, -0.4653, -0.0958,  0.6321, -0.0694,  0.2954, -0.0730,        
#          1.3622,  0.6438,  1.5559,  0.8432])
# z = x.view(5,-1) #-1 => 나머지는 알아서 계산(전체 숫자에 대해서)
# print(z)
# tensor([[ 0.6721, -0.7483,  0.7681,  0.2374],
#         [-0.6830,  1.5633,  1.2590, -0.7781],
#         [-0.2972, -0.3517, -0.4653, -0.0958],
#         [ 0.6321, -0.0694,  0.2954, -0.0730],
#         [ 1.3622,  0.6438,  1.5559,  0.8432]])

# item : 텐서의 값이 단 하나라도 존재한다면 숫자값을 얻을 수 있다.
# 텐서의 값 하나를 가져옴
# 스칼라 값이 2개 이상이면 error, 무엇을 가져와야할지 모름
# x = torch.rand(1)
# print(x) #tensor([0.4723])
# print(x.item()) #0.472270131111145 #tensor에서는 값이 축약됨
# print(x.dtype) #torch.float32

# squeeze : 차원을 축소(제거)
# tensor = torch.rand(1,3,3)
# print(tensor)
# tensor([[[0.9955, 0.9276, 0.7738],
#          [0.0408, 0.7112, 0.0032],
#          [0.4841, 0.7243, 0.1597]]])
# print(tensor.shape) #torch.Size([1, 3, 3])
# t = tensor.squeeze()
# print(t)
# tensor([[0.9955, 0.9276, 0.7738],
#         [0.0408, 0.7112, 0.0032],
#         [0.4841, 0.7243, 0.1597]])
# print(t.shape) #torch.Size([3, 3])

# unsqueeze : 차원을 증가(생성)
# t = torch.rand(3,3)
# print(t) 
# tensor([[0.2225, 0.8936, 0.9114],
#         [0.9487, 0.2216, 0.1833],
#         [0.2503, 0.3414, 0.8757]])
# print(t.shape) #torch.Size([3, 3])
# t = t.unsqueeze(dim=0) #dim =0 == 첫번째 차원
# print(t)
# tensor([[[0.2225, 0.8936, 0.9114],
#          [0.9487, 0.2216, 0.1833],
#          [0.2503, 0.3414, 0.8757]]])
# print(t.shape) #torch.Size([1, 3, 3])

# w = torch.rand(3,3)
# print(w)
# tensor([[0.9820, 0.9939, 0.2950],
#         [0.7026, 0.5443, 0.9728],
#         [0.4744, 0.1327, 0.0796]])
# print(w.shape) #torch.Size([3, 3])
# w = w.unsqueeze(dim=2)
# print(w)
# tensor([[[0.9820],
#          [0.9939],
#          [0.2950]],

#         [[0.7026],
#          [0.5443],
#          [0.9728]],

#         [[0.4744],
#          [0.1327],
#          [0.0796]]])
# print(w.shape) #torch.Size([3, 3, 1])

# stack : 텐서간 결합
# x = torch.FloatTensor([1,4])
# print(x) #tensor([1., 4.])
# y = torch.FloatTensor([2,5])
# print(y) #tensor([2., 5.])
# z = torch.FloatTensor([3,6])
# print(z) #tensor([3., 6.])
# print(torch.stack([x,y,z]))
# tensor([[1., 4.],
#         [2., 5.],
#         [3., 6.]])

# cat : 텐서를 결합하는 메소드
# 넘파이의 stack과 유사, 쌓을 dim이 존재
# 해당 차원을 늘려준 후 결합
# a = torch.rand(1,3,3)
# print(a)
# tensor([[[0.7397, 0.3503, 0.3361],
#          [0.4840, 0.6325, 0.2693],
#          [0.0698, 0.4965, 0.0430]]])
# b = torch.rand(1,3,3)
# print(b)
# tensor([[[0.2951, 0.8212, 0.1773],
#          [0.8734, 0.7483, 0.7379],
#          [0.6148, 0.6687, 0.1134]]])
# c = torch.cat((a,b), dim = 0)
# print(c)
# tensor([[[0.7397, 0.3503, 0.3361],
#          [0.4840, 0.6325, 0.2693],
#          [0.0698, 0.4965, 0.0430]],

#         [[0.2951, 0.8212, 0.1773],
#          [0.8734, 0.7483, 0.7379],
#          [0.6148, 0.6687, 0.1134]]])
# print(c.size()) #torch.Size([2, 3, 3])
# c = torch.cat((a,b), dim = 1)
# print(c)
# tensor([[[0.3013, 0.1155, 0.1550],
#          [0.1104, 0.5779, 0.0019],
#          [0.7206, 0.8821, 0.3019],
#          [0.1936, 0.1309, 0.7398],
#          [0.3730, 0.4482, 0.4588],
#          [0.9112, 0.0235, 0.2236]]])
# print(c.size()) #torch.Size([1, 6, 3])
# c= torch.cat((a,b), dim = 2)
# print(c)
# tensor([[[0.2155, 0.9125, 0.2364, 0.3966, 0.7667, 0.3982],
#          [0.5560, 0.9339, 0.3878, 0.6583, 0.4329, 0.1527],
#          [0.0147, 0.4168, 0.7318, 0.7820, 0.3263, 0.6221]]])
# print(c.size()) #torch.Size([1, 3, 6])

# chunk : 텐서를 여러 개로 나눌 때 사용(몇 개로 나눌 것인가?)
# tensor = torch.rand(3,6)
# print(tensor)
# tensor([[0.9027, 0.9262, 0.7457, 0.7135, 0.2129, 0.9219],
#         [0.7726, 0.2888, 0.8910, 0.4776, 0.8088, 0.3700],
#         [0.7131, 0.9451, 0.3759, 0.1273, 0.1328, 0.9741]])
# t1, t2, t3 = torch.chunk(tensor,3,dim=1)
# print(t1)
# tensor([[0.9027, 0.9262],
#         [0.7726, 0.2888],
#         [0.7131, 0.9451]])
# print(t2)
# tensor([[0.7457, 0.7135],
#         [0.8910, 0.4776],
#         [0.3759, 0.1273]])
# print(t3)
# tensor([[0.2129, 0.9219],
#         [0.8088, 0.3700],
#         [0.1328, 0.9741]])

# split : chunk와 동일 기능,(텐서의 크기는 몇 인가?)
# tensor = torch.rand(3,6)
# t1, t2 = torch.split(tensor,3,dim=1)
# print(tensor)
# tensor([[0.1657, 0.1830, 0.5426, 0.4667, 0.0295, 0.6834],
#         [0.5543, 0.4771, 0.6407, 0.6803, 0.4836, 0.6334],
#         [0.2760, 0.9154, 0.3663, 0.3247, 0.0619, 0.5830]])
# print(t1)
# tensor([[0.1657, 0.1830, 0.5426],
#         [0.5543, 0.4771, 0.6407],
#         [0.2760, 0.9154, 0.3663]])
# print(t2)
# tensor([[0.4667, 0.0295, 0.6834],
#         [0.6803, 0.4836, 0.6334],
#         [0.3247, 0.0619, 0.5830]])
