#include <cuda.h>
#include <iostream>

__global__ void addOne(int *a, int n) {
    int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadID < n) {
        a[threadID] += 1;
    }
}

void printArray(int *a, int n) {
    for (int i = 0; i < n; ++i) {
        std::cout << "Element at index " << i << ": " << a[i] << std::endl;
    }
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0); // 获取第0个设备的信息
        std::cout << "Running on GPU: " << prop.name << std::endl;
    }

    const long long int N = 1000000000; // 调整数组大小到一个合理的值，以免超出GPU显存限制
    int *h_a, *d_a;

    // 在主机上分配和初始化内存
    h_a = new int[N];
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
    }

    // 在设备（GPU）上分配内存
    cudaMalloc((void**)&d_a, N * sizeof(int));

    // 将主机数据复制到设备
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);

    // 循环启动CUDA内核函数
    for (int i = 0; i < 10000; ++i) {
        // 启动CUDA内核函数
        addOne<<<1, N>>>(d_a, N);
    }

    // 确保所有CUDA kernel都已完成执行
    cudaDeviceSynchronize();

    // 将设备上的结果复制回主机
    cudaMemcpy(h_a, d_a, N * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印主机上的结果
    printArray(h_a, N);

    // 释放内存
    cudaFree(d_a);
    delete[] h_a;

    return 0;
}
