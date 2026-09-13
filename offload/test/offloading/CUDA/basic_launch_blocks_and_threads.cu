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

__global__ void incrementCounter(int *A) {
  __scoped_atomic_fetch_add(A, 1, __ATOMIC_SEQ_CST, __MEMORY_SCOPE_DEVICE);
}

int main(int argc, char **argv) {
  int *Ptr, I;
  cudaMalloc(&Ptr, sizeof(int));
  printf("Ptr %p\n", Ptr);
  // CHECK: Ptr [[Ptr:0x.*]]
  int Zero = 0;
  cudaMemcpy(Ptr, &Zero, sizeof(int), cudaMemcpyHostToDevice);
  incrementCounter<<<7, 6>>>(Ptr);
  cudaDeviceSynchronize();
  cudaMemcpy(&I, Ptr, sizeof(int), cudaMemcpyDeviceToHost);
  printf("I: %i\n", I);
  // CHECK: I: 42
}
