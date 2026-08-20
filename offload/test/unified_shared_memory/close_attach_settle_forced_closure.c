// RUN: %libomptarget-compile-run-and-check-generic

// REQUIRES: unified_shared_memory
// UNSUPPORTED: clang-6, clang-7, clang-8, clang-9

// amdgpu runtime crash
// Fails on nvptx with error: an illegal memory access was encountered
// UNSUPPORTED: amdgcn-amd-amdhsa
// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: intelgpu

// The obligation to hold device addresses propagates along a chain, so it has to
// be worked out transitively before any storage is changed.
//
//   a -> b -> x
//
// x was allocated by an earlier construct, so it cannot give that allocation up,
// which means b must hold a device address. b is allocated by this construct, so
// it would otherwise be a candidate for release on account of a -- but releasing
// it would reintroduce the very conflict on x that cannot be resolved. b is
// reachable from x, so it has to be left alone and a upgraded instead.
//
// Looking only at attachments whose pointee is immediately unreleasable misses
// this, because b starts out device-bound and so the a -> b attachment looks
// like an ordinary candidate for release.

#include <omp.h>
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

  // Allocated here, so the construct below cannot release it.
#pragma omp target enter data map(close, to : x)

#pragma omp target data map(close, to : b) map(alloc : a)                      \
    map(alloc : b[0 : 1], a[0 : 1])
  {
  }

  // CHECK: a == &b, b == &x
  printf("a %s &b, b %s &x\n", a == &b ? "==" : "!=", b == &x ? "==" : "!=");

  // CHECK: Done!
  printf("Done!\n");
  return 0;
}
