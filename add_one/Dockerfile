# 使用NVIDIA CUDA官方基础镜像（这里以Ubuntu 20.04 + CUDA 11为例）
# FROM nvidia/cuda:11.1.1-devel-ubuntu20.04
FROM 192.168.70.202:32373/registry/nvidia/cuda:11.1.1-devel-ubuntu20.04  

# 更新系统并安装必要的编译工具
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git && \
    rm -rf /var/lib/apt/lists/*

# 将源码复制到容器内部的工作目录
COPY cuda_add_one.cu /app/

# 编译CUDA程序
RUN nvcc -o /app/add_one /app/cuda_add_one.cu

# 设置工作目录
WORKDIR /app

# 运行CUDA程序
CMD ["./add_one"]