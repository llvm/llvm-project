// RUN: %libomptarget-compile-run-and-check-generic

// REQUIRES: unified_shared_memory
// UNSUPPORTED: clang-6, clang-7, clang-8, clang-9

// amdgpu runtime crash
// Fails on nvptx with error: an illegal memory access was encountered
// UNSUPPORTED: amdgcn-amd-amdhsa
// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: intelgpu

#pragma omp requires unified_shared_memory

#include <stdio.h>

#pragma omp begin declare target
double base[10] = {0};
#pragma omp end declare target

int main() {

// close range covers base[1] and should properly
// update the budder from the outer mapping.
#pragma omp target map(close, alloc : base[1 : 9])
  {
    base[1] = 99.0;
  }
#pragma omp target update from(base)
  // CHECK: base[1] = 99.000000 (expected 99.0)
  printf("base[1] = %f (expected 99.0)\n", base[1]);
  return 0;
}