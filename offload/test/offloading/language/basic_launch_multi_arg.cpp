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

// REQUIRES: gpu
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: amdgcn-amd-amdhsa-LTO
// UNSUPPORTED: amdgpu-amd-amdhsa-LTO
// UNSUPPORTED: intelgpu

// clang-format off
#include <stdio.h>
#include "Inputs/DefineTestLanguageNames.inc"
// clang-format on

__global__ void square(int *Dst, short Q, int *Src, short P) {
  *Dst = (Src[0] + Src[1]) * (Q + P);
  Src[0] = Q;
  Src[1] = P;
}

int main(int argc, char **argv) {
  int *Src, *Ptr;
  Malloc(&Ptr, 4);
  Malloc(&Src, 8);

  int I = 7;
  int HostSrc[2] = {-2, 8};
  Memcpy(Ptr, &I, sizeof(int), MemcpyHostToDevice);
  Memcpy(Src, &HostSrc[0], 2 * sizeof(int), MemcpyHostToDevice);
  square<<<1, 1>>>(Ptr, 3, Src, 4);
  DeviceSynchronize();
  Memcpy(&I, Ptr, sizeof(int), MemcpyDeviceToHost);
  Memcpy(&HostSrc[0], Src, 2 * sizeof(int), MemcpyDeviceToHost);
  printf("I: %i\n", I);
  // CHECK: I: 42
  printf("Src: %i, %i\n", HostSrc[0], HostSrc[1]);
  // CHECK: Src: 3, 4
}
