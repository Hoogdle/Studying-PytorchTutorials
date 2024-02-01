### {Pytorch}

# 초기 Numpy와 유사한 과학 연산 라이브러리
# 후기 딥러닝 프레임워크로 발전
# GPU ==> 병렬처리 가능
# 파이썬을 닮음

## {Pytorch 구조}

# https://youtu.be/k60oT_8lyFw?list=PL7ZVZgsnLwEEIC4-KQIchiPda_EjxX61r&t=189
# C, Cuda(하위레벨) => C++(중간레벨) => Python(상위레벨)
# Python에서 다양한 모듈이 동작하도록 함

## {Pytorch 구성요소}
# torch : 메인 네임스페이스, tensor 및 다양한 수학 함수 
# torch.autograd : 자동 미분 기능
# torch.nn : 신경망 구축
# torch.multiprocessing : 병렬처리 기능
# torch.optim : SGD 중심, 파라미터 최적화 알고리즘
# torch.utils : 데이터 조작, 유틸리티 제공
# torch.onnx : ONNX(Open Neural Netword Exchange) 타 프레임워크 모델 공유

### {Tensors - 텐서}
# 데이터 표현을 위한 기본구조
# 데이터를 담기 위한 Container
# 다차원의 데이터 표현을 위해 사용
# 수치형 데이터 저장
# NumPyu의 ndarray와 유사
# GPU 사용 => 연산 가속 가능
# https://youtu.be/k60oT_8lyFw?list=PL7ZVZgsnLwEEIC4-KQIchiPda_EjxX61r&t=420

# 0D Tensor == Scalar, Rank:0 , Shape() #
# 1D Tesnor == Vector, Rank:1 , Shape(3,), 1x3
# 2D Tensor == Matrix, Rank:2 , Shape(3,3) 3x3 
# 3D Tensor, Rank:3, Shape(3,3,3) 3x3x3
# Over 4D, 3차원 박스를 기준으로 다룸 # (3x3x3) == 길이3, 3D 박스
# 4D Tensor, Rank:4, Shape(3,3,3,3) (3x3x3)x3 
# 5D Tensor, Rank:5, Shape(3,3,3,3,3) (3x3x3)x(3x3)
# 6D Tensor, Rank:6, Shape(3,3,3,3,3,3) (3x3x3)x(3x3x3)

## import torch 로 들여옴
import torch

## Tensor 초기화, DataType

# 1. 초기화 되지 않은 텐서
# 4x2 초기화 되지 않은 tensor
# x = torch.empty(4,2)
# print(x)
# tensor([[9.6034e+01, 1.8890e-42],
#         [0.0000e+00, 0.0000e+00],
#         [0.0000e+00, 0.0000e+00],
#         [0.0000e+00, 0.0000e+00]])

# 2. 무작위로 초기화된 텐서
# 4x2 random 초기화 된 텐서
# x = torch.rand(4,2)
# print(x)
# tensor([[0.2935, 0.6161],
#         [0.0289, 0.4697],
#         [0.1332, 0.9857],
#         [0.3243, 0.8467]])

# 3. Datatype = long, 0으로 채워진 Tensor
# x = torch.zeros(4,2,dtype=torch.long)
# print(x)
# tensor([[0, 0],
#         [0, 0],
#         [0, 0],
#         [0, 0]])

# 4. 사용자 입력값으로 텐서 초기화
# 기본적으로 실수형으로 들어감
# x = torch.tensor([3,2.3])
# print(x)
# tensor([3.0000, 2.3000])

# 5. 2x4, double type, 1로 채움
# x = torch.ones(2,4,dtype=torch.double)
# print(x)
# tensor([[1., 1., 1., 1.],
#         [1., 1., 1., 1.]], dtype=torch.float64)

# 기존 Tensor로 새로운 Tensor
# x = x.new_zeros(2,4,dtype = torch.int)
# print(x)
# tensor([[0, 0, 0, 0],
#         [0, 0, 0, 0]], dtype=torch.int32)

# 6. x와 같은 크기, float 타입, 무작위로 채워진 텐서
# _like(x) 기존 x텐서와 '같은 모양' 으로 생성
# x = torch.rand_like(x, dtype=torch.float)
# print(x)
# tensor([[0.6403, 0.8389, 0.8881, 0.4158],
#         [0.3337, 0.4814, 0.9622, 0.9827]])

# 7. 텐서의 크기 계산
# print(x.size()) #torch.Size([2])

## 데이터 타입(Data Type)

# FloatTensor ==> float32 타입의 텐서
# ft = torch.FloatTensor([1,2,3])
# print(ft) #tensor([1., 2., 3.])
# print(ft.dtype) #torch.float32

# 데이터 타입 변경
# print(ft.short()) #tensor([1, 2, 3], dtype=torch.int16)
# print(ft.int()) #tensor([1, 2, 3], dtype=torch.int32)
# print(ft.long()) #tensor([1, 2, 3])

# it = torch.IntTensor([1,2,3])
# print(it) #tensor([1, 2, 3], dtype=torch.int32)
# print(it.dtype) #torch.int32

# print(it.float()) #tensor([1., 2., 3.])
# print(it.double()) #tensor([1., 2., 3.], dtype=torch.float64)
# print(it.half()) #tensor([1., 2., 3.], dtype=torch.float16)

## CUDA Tensors
# .to => GPU or CPU 로 move

# x = torch.rand(1) #랜덤하게 1개
# print(x) #tensor([0.7177])
# print(x.item()) #0.7176821827888489
# print(x.dtype) #torch.float32

# torch device, cuda 를 넣으면 cuda, cpu를 넣으면 cpu 로 동작
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device) #cuda
# y = torch.ones_like(x, device = device)
# print(y) #tensor([1.], device='cuda:0')
# x = x.to(device) # x를 'cuda'로 보냄
# print(x) #tensor([0.9205], device='cuda:0')
# z = x+y 
# print(z) #tensor([1.9205], device='cuda:0')
# print(z.to('cpu',torch.double)) #tensor([1.9205], dtype=torch.float64)

## 다차원 Tensor 표현

# 0D Tensor(Sclar)
# 하나의 숫자를 담고 있는 텐서(tensor)
# 축과 형상이 없음
# t0 = torch.tensor(0)
# print(t0.ndim) #0
# print(t0.shape) #torch.Size([])
# print(t0) #tensor(0)

# 1D Tensor(Vector)
# 값들을 저장한 리스트와 유사한 텐서
# 하나의 축이 존재
# t1 = torch.tensor([1,2,3])
# print(t1.ndim) #1
# print(t1.shape) #torch.Size([3])
# print(t1) #tensor([1, 2, 3])

# 2D Tensor(Matrix)
# 행렬과 같은 모양, 두 개의 축이 존재
# 일반적이 수치, 통계 데이터셋이 해당
# 주로 샘플(sample)과 특성(faetures)를 가진 구조로 사용
# t2 = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
# print(t2.ndim) #2
# print(t2.shape) #torch.Size([3, 3])
# print(t2) #tensor([[1, 2, 3],
        # [4, 5, 6],
        # [7, 8, 9]])

# 3D Tensor
# Cube와 같은 모양, 세 개의 축 존재
# 데이터가 연속된 시퀀스 데이터 or 시간 축이 포함된 시계열 데이터에 해당
# 주식 가격 데이터셋, 시간에 따른 질병 발병 데이터 등이 존재
# 주로 샘플(samples),타임스텝(timesteps),특성(features)을 가진 구조로 사용
# t3 = torch.tensor([[[1,2,3],[4,5,6],[7,8,9]],
#                   [[1,2,3],[4,5,6],[7,8,9]],
#                   [[1,2,3],[4,5,6],[7,8,9]]])
# print(t3.ndim) #3
# print(t3.shape) #torch.Size([3, 3, 3])
# print(t3) #tensor([[[1, 2, 3],
        #  [4, 5, 6],
        #  [7, 8, 9]],

        # [[1, 2, 3],
        #  [4, 5, 6],
        #  [7, 8, 9]],

        # [[1, 2, 3],
        #  [4, 5, 6],
        #  [7, 8, 9]]])

# 4D Tensor
# 4개의 축
# 컬러 이미지 데이터가 대표 사례(흑백 이미지는 3D Tensor로 가능)
# 주로 샘플(samples),높이(height),너비(width),컬러 채널(channel)을 가진 구조

# 5D Tensor
# 5개의 축
# 비디오 데이터가 대표 사례
# 주로 샘플(samples), 프레임(frames), 높이(height), 너비(width), 컬러 채널(channel)을 가진 구조로 사용


