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
#include <omp.h>
#include <stdio.h>

int main() {
  int x = 111;

  // CHECK: present when unmapped: 0
  printf("present when unmapped: %d\n",
         omp_target_is_present(&x, omp_get_default_device()));

#pragma omp target_enter_data map(alloc : x)

  // CHECK: present after mapping: 1
  printf("present after mapping: %d\n",
         omp_target_is_present(&x, omp_get_default_device()));
  return 0;
}