// clang-format off
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native -x cuda -DOFFLOAD_TEST_LANGUAGE=cuda %s -o %t.cuda
// RUN: %t.cuda | %fcheck-generic
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native -x cuda -DOFFLOAD_TEST_LANGUAGE=cuda %s -o %t.cuda.omp -fopenmp
// RUN: %t.cuda.omp | %fcheck-generic
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native -x hip -DOFFLOAD_TEST_LANGUAGE=hip %s -o %t.hip
// RUN: %t.hip | %fcheck-generic
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native -x hip -DOFFLOAD_TEST_LANGUAGE=hip %s -o %t.hip.omp -fopenmp
// RUN: %t.hip.omp | %fcheck-generic
// clang-format on

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: amdgcn-amd-amdhsa-LTO
// UNSUPPORTED: amdgpu-amd-amdhsa-LTO
// UNSUPPORTED: intelgpu

// clang-format off
#include <stdio.h>
#include "Inputs/DefineTestLanguageNames.inc"
// clang-format on

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
  Malloc(&DevPtr, sizeof(int));
  reduceBlock<<<1, 64>>>(DevPtr);
  DeviceSynchronize();
  Memcpy(&Result, DevPtr, sizeof(int), MemcpyDeviceToHost);

  printf("sum: %i\n", Result);
  // CHECK: sum: 2016
}
