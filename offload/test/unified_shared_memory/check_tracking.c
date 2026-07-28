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

#pragma omp begin declare target
int x = 111;
#pragma omp end declare target
int y = 111;

int present(void *p) {
  return omp_target_is_present(p, omp_get_default_device());
}

int main() {
  int xl = 111;

  // CHECK: present when unmapped: 0
  printf("present when unmapped: %d\n", present(&xl));

#pragma omp target_enter_data map(alloc : xl)

  // CHECK: present after mapping: 1
  printf("present after mapping: %d\n", present(&xl));
#pragma omp target_exit_data map(from : xl)
  // CHECK: present after mapping: 0
  printf("present after mapping: %d\n", present(&xl));

  // CHECK: present when unmapped: 1 0
  printf("present when unmapped: %d %d\n", present(&x), present(&y));

#pragma omp target_enter_data map(to : x, y)

  // CHECK: present after mapping: 1 1
  printf("present after mapping: %d %d\n", present(&x), present(&y));

#pragma omp target_exit_data map(from : x, y)

  // CHECK: present after mapping: 1 0
  printf("present after mapping: %d %d\n", present(&x), present(&y));
  return 0;
}