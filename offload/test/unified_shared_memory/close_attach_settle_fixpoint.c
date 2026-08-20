// RUN: %libomptarget-compile-run-and-check-generic

// REQUIRES: unified_shared_memory
// UNSUPPORTED: clang-6, clang-7, clang-8, clang-9

// amdgpu runtime crash
// Fails on nvptx with error: an illegal memory access was encountered
// UNSUPPORTED: amdgcn-amd-amdhsa
// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: intelgpu

// Releasing one pointee's device allocation can leave a second attachment
// unsettled, because the storage just released may itself hold a pointer whose
// own pointee is device-bound. The releases therefore have to be repeated until
// none is left to do, before any pointer is given device storage.
//
// Here a is host-bound and already present, so it can never hold a device
// address, while b and its pointee are both newly mapped with close:
//
//   a -> b -> x
//
// Releasing b for the sake of a's attachment leaves b host-bound, which in turn
// makes b's own attachment to the device-bound x unsettled. Releasing x as well
// settles everything with all three host-bound. Considering the releases only
// once would instead give b device storage again and then fail on a, reporting a
// conflict for a configuration that has a solution.

#include <stdio.h>

#pragma omp requires unified_shared_memory

int x = 9;
int *b = &x;
int **a = &b;

int main(void) {
  // The unified_shared_memory requirement is registered when a device image is
  // loaded, so the program needs a target region for it to take effect.
#pragma omp target
  {
  }

  // The pre-existing host-bound mapping that cannot be changed later.
#pragma omp target enter data map(alloc : a)

  // Attaches b to x, and a to b.
#pragma omp target data map(close, to : b) map(close, to : b[0 : 1])           \
    map(alloc : a[0 : 1])
  {
  }

  // CHECK: a == &b, b == &x
  printf("a %s &b, b %s &x\n", a == &b ? "==" : "!=", b == &x ? "==" : "!=");

  // CHECK: Done!
  printf("Done!\n");
  return 0;
}
