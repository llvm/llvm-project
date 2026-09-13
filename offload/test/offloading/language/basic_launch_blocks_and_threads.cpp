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

__global__ void incrementCounter(int *A) {
  __scoped_atomic_fetch_add(A, 1, __ATOMIC_SEQ_CST, __MEMORY_SCOPE_DEVICE);
}

int main(int argc, char **argv) {
  int *Ptr, I;
  Malloc(&Ptr, sizeof(int));
  printf("Ptr %p\n", Ptr);
  // CHECK: Ptr [[Ptr:0x.*]]
  Memset(Ptr, 0, sizeof(int));
  incrementCounter<<<7, 6>>>(Ptr);
  DeviceSynchronize();
  Memcpy(&I, Ptr, sizeof(int), MemcpyDeviceToHost);
  printf("I: %i\n", I);
  // CHECK: I: 42
}
