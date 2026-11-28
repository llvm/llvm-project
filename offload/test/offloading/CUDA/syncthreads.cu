// clang-format off
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native %s -o %t
// RUN: %t | %fcheck-generic
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native %s -o %t -fopenmp
// RUN: %t | %fcheck-generic
// clang-format on

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: amdgcn-amd-amdhsa-LTO
// UNSUPPORTED: amdgpu-amd-amdhsa-LTO
// UNSUPPORTED: intelgpu

#include <stdio.h>

__global__ void reduceBlock(int *Out) {
  __shared__ int Scratch[64];
  int Tid = threadIdx.x;
  Scratch[Tid] = Tid;
  __syncthreads();

  if (Tid == 0) {
    int Sum = 0;
    for (int I = 0; I < 64; ++I)
      Sum += Scratch[I];
    Out[0] = Sum;
  }
}

int main(int argc, char **argv) {
  int *DevPtr;
  int Result = 0;
  cudaMalloc(&DevPtr, sizeof(int));
  reduceBlock<<<1, 64>>>(DevPtr);
  cudaDeviceSynchronize();
  cudaMemcpy(&Result, DevPtr, sizeof(int), cudaMemcpyDeviceToHost);

  printf("sum: %i\n", Result);
  // CHECK: sum: 2016
}
