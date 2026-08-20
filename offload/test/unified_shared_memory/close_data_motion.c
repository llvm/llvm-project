// RUN: %libomptarget-compile-run-and-check-generic

// REQUIRES: unified_shared_memory
// UNSUPPORTED: clang-6, clang-7, clang-8, clang-9

// amdgpu runtime crash
// Fails on nvptx with error: an illegal memory access was encountered
// UNSUPPORTED: amdgcn-amd-amdhsa
// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: intelgpu

// A close map under unified shared memory gets its own device allocation, so it
// must retain normal device data-motion semantics: values written on the device
// have to be copied back for a `from` map, and `target update` on such a
// mapping must actually transfer data rather than being treated as a no-op.
//
// This is easy to break by treating every mapping under USM as a host pointer,
// because the copy-back and the update path are both skipped for host pointers.
// When that happens the device writes are silently lost.

#include <stdio.h>

#pragma omp requires unified_shared_memory

#define N 64

int main() {
  int a[N], b[N];

  for (int i = 0; i < N; ++i) {
    a[i] = 1;
    b[i] = 1;
  }

  // The device gets its own copy of "a" because of close. The writes below must
  // make it back to the host at the end of the region.
#pragma omp target map(close, tofrom : a[ : N])
  {
    for (int i = 0; i < N; ++i)
      a[i] += 10;
  }

  int fails = 0;
  for (int i = 0; i < N; ++i)
    if (a[i] != 11)
      fails++;
  // CHECK: close tofrom copied back: Succeeded
  printf("close tofrom copied back: %s\n",
         (fails == 0) ? "Succeeded" : "Failed");

  // Same, but the data motion is requested explicitly with target update.
#pragma omp target data map(close, alloc : b[ : N])
  {
    // Push the current host values into the device copy.
#pragma omp target update to(b[ : N])

#pragma omp target map(present, alloc : b[ : N])
    {
      for (int i = 0; i < N; ++i)
        b[i] += 20;
    }

    // Pull the device values back out. If update is a no-op the host keeps 1.
#pragma omp target update from(b[ : N])
  }

  fails = 0;
  for (int i = 0; i < N; ++i)
    if (b[i] != 21)
      fails++;
  // CHECK: close target update from: Succeeded
  printf("close target update from: %s\n",
         (fails == 0) ? "Succeeded" : "Failed");

  // CHECK: Done!
  printf("Done!\n");
  return 0;
}
