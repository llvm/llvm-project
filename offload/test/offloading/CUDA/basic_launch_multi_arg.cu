// clang-format off
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native %s -o %t
// RUN: %t | %fcheck-generic
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native %s -o %t -fopenmp 
// RUN: %t | %fcheck-generic
// clang-format on

// REQUIRES: gpu
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: amdgcn-amd-amdhsa-LTO
// UNSUPPORTED: amdgpu-amd-amdhsa-LTO
// UNSUPPORTED: intelgpu

#include <stdio.h>

__global__ void square(int *Dst, short Q, int *Src, short P) {
  *Dst = (Src[0] + Src[1]) * (Q + P);
  Src[0] = Q;
  Src[1] = P;
}

int main(int argc, char **argv) {
  int *Src, *Ptr;
  cudaMalloc(&Ptr, 4);
  cudaMalloc(&Src, 8);

  int I = 7;
  int HostSrc[2] = {-2, 8};
  cudaMemcpy(Ptr, &I, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(Src, &HostSrc[0], 2 * sizeof(int), cudaMemcpyHostToDevice);
  square<<<1, 1>>>(Ptr, 3, Src, 4);
  cudaDeviceSynchronize();
  cudaMemcpy(&I, Ptr, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&HostSrc[0], Src, 2 * sizeof(int), cudaMemcpyDeviceToHost);
  printf("I: %i\n", I);
  // CHECK: I: 42
  printf("Src: %i, %i\n", HostSrc[0], HostSrc[1]);
  // CHECK: Src: 3, 4
}
