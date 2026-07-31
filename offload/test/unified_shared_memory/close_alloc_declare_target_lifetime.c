// RUN: %libomptarget-compile-run-and-check-generic

// REQUIRES: unified_shared_memory
// UNSUPPORTED: clang-6, clang-7, clang-8, clang-9

// amdgpu runtime crash
// Fails on nvptx with error: an illegal memory access was encountered
// UNSUPPORTED: amdgcn-amd-amdhsa
// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: intelgpu

// A declare-target variable is in the device data environment for the whole
// program, so its storage outlives any individual mapping of it, and a
// `target update` on it after some unrelated region has ended is valid user
// code.
//
// This test documents that the implementation-specific optimization of giving a
// `close` mapping its own device allocation conflicts with that: with
// map(close, alloc : ...) there is no copy-back and the separate allocation is
// released at the end of the region, so a value the device wrote there is lost.
//
// The optimization applies only to storage that is not already on the device. A
// mapping that lies within such storage stays on the host path and so shares
// its device buffer, which is what the containment check in getTargetPointer()
// expresses. For a declare-target variable that requires knowing the variable's
// extent, which code generation communicates in its offload entry: under
// unified shared memory the entry otherwise describes only the device reference
// pointer.

#include <stdio.h>

#pragma omp requires unified_shared_memory

#pragma omp begin declare target
double part[10] = {0};
double viaptr[10] = {0};
#pragma omp end declare target

// An unrelated variable, mapped alongside the close mapping below. Its entry is
// what makes the lookup for the close mapping find a neighbor, so that the
// mapping is not left on the host path.
double *probe;

int main() {
  // (A) A close mapping of part of a declare-target object.
  //
  // CHECK: A: 11.000000
#pragma omp target map(close, alloc : part[3 : 4]) map(from : probe)
  {
    probe = &part[3];
    part[3] = 11.0;
  }
  // Legal: "part" is still in the device data environment.
#pragma omp target update from(part[3 : 1])
  printf("A: %f\n", part[3]);

  // (B) The same, reached through a pointer into the middle of the object, so
  // the mapping carries no reference to the declare-target variable at all.
  //
  // CHECK: B: 22.000000
  double *p = &viaptr[6];
#pragma omp target map(close, alloc : p[0 : 3]) map(from : probe)
  {
    probe = &p[0];
    p[0] = 22.0;
  }
#pragma omp target update from(viaptr[6 : 1])
  printf("B: %f\n", viaptr[6]);

  // CHECK: Done!
  printf("Done!\n");
  return 0;
}
