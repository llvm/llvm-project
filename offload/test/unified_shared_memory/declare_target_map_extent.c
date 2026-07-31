// RUN: %libomptarget-compile-run-and-check-generic

// REQUIRES: unified_shared_memory
// UNSUPPORTED: clang-6, clang-7, clang-8, clang-9

// amdgpu runtime crash
// Fails on nvptx with error: an illegal memory access was encountered
// UNSUPPORTED: amdgcn-amd-amdhsa
// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: intelgpu

// Mapping a declare-target aggregate, or a subsection of one that starts at its
// beginning, must work under unified shared memory.
//
// If the runtime registers a mapping for the storage of a declare-target
// variable, the extent it registers has to be the extent of the variable. Using
// the size of the offload entry instead describes the device reference pointer,
// i.e. sizeof(void *), and then a map of the real object looks like an attempt
// to extend an existing, smaller mapping, which is rejected:
//
//   explicit extension not allowed: host address specified is ... (80 bytes),
//   but device allocation maps to host at ... (8 bytes)
//
// That aborts the program rather than producing a wrong value, and it happens
// for a plain map as well as for a close one.

#include <stdio.h>

#pragma omp requires unified_shared_memory

#pragma omp begin declare target
double base[10] = {0};
#pragma omp end declare target

int main() {
  // A plain map of the whole declare-target array.
#pragma omp target map(tofrom : base[0 : 10])
  {
    base[0] = 7.0;
  }
  // CHECK: plain whole-array map: base[0] = 7.000000
  printf("plain whole-array map: base[0] = %f\n", base[0]);

  // A close map of the whole array.
#pragma omp target map(close, alloc : base[0 : 10])
  {
    base[3] = 5.0;
  }
#pragma omp target update from(base)
  // CHECK: close whole-array map: base[3] = 5.000000
  printf("close whole-array map: base[3] = %f\n", base[3]);

  // A close map of a subsection that starts at the beginning of the array, so
  // it overlaps whatever was registered for the variable itself.
#pragma omp target map(close, alloc : base[0 : 9])
  {
    base[1] = 99.0;
  }
#pragma omp target update from(base)
  // CHECK: close leading subsection: base[1] = 99.000000
  printf("close leading subsection: base[1] = %f\n", base[1]);

  // CHECK: Done!
  printf("Done!\n");
  return 0;
}
