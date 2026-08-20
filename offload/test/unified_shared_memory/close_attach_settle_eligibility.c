// RUN: %libomptarget-compile-run-and-check-generic

// REQUIRES: unified_shared_memory
// UNSUPPORTED: clang-6, clang-7, clang-8, clang-9

// amdgpu runtime crash
// Fails on nvptx with error: an illegal memory access was encountered
// UNSUPPORTED: amdgcn-amd-amdhsa
// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: intelgpu

// An attachment is only performed when one of the two sides was newly mapped by
// the construct that prescribes it. Deciding which side holds device storage has
// to apply that same test: a construct that maps neither side afresh performs no
// assignment, so there is no conflict to resolve even when the pointer is
// host-bound and the pointee device-bound.
//
// Here p and x are both mapped by earlier constructs, so the map of the
// zero-length array section prescribes nothing and must not be reported as an
// unsatisfiable attachment.

#include <stdio.h>

#pragma omp requires unified_shared_memory

int x = 7;
int *p = &x;

int main(void) {
  // The unified_shared_memory requirement is registered when a device image is
  // loaded, so the program needs a target region for it to take effect.
#pragma omp target
  {
  }

  // p is host-bound, x is device-bound, and both are already present.
#pragma omp target enter data map(alloc : p)
#pragma omp target enter data map(close, to : x)

#pragma omp target data map(alloc : p[0 : 1])
  {
  }

  // CHECK: p == &x
  printf("p %s &x\n", p == &x ? "==" : "!=");

  // CHECK: Done!
  printf("Done!\n");
  return 0;
}
