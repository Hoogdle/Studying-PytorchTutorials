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

## {Tensors - 텐서}
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

# import torch 로 들여옴
import torch

# Tensor 초기화, DataType

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

# https://youtu.be/k60oT_8lyFw?list=PL7ZVZgsnLwEEIC4-KQIchiPda_EjxX61r&t=778