// RUN: %libomp-compile
// RUN: env OMP_ALLOCATOR=111 %libomp-run 2>&1 | FileCheck %s
// RUN: env OMP_ALLOCATOR=omp_default_mem_alloc_xyz %libomp-run 2>&1 | FileCheck %s
// UNSUPPORTED: gcc

// Both invocations of the test should produce (different) warnings:
// OMP: Warning #42: OMP_ALLOCATOR: "111" is an invalid value; ignored.
// OMP: Warning #189: Allocator omp_const_mem_alloc is not available, will use default allocator.
#include <stdio.h>
#include <omp.h>
int main() {
  volatile int n = omp_get_max_threads(); // causes library initialization
  return 0;
}

// CHECK: {{^OMP: Warning #[0-9]+}}: {{.*$}}
