// RUN: %libomptarget-compile-generic
// RUN: %libomptarget-run-fail-generic 2>&1 | %fcheck-generic

// REQUIRES: unified_shared_memory
// UNSUPPORTED: clang-6, clang-7, clang-8, clang-9

// amdgpu runtime crash
// Fails on nvptx with error: an illegal memory access was encountered
// UNSUPPORTED: amdgcn-amd-amdhsa
// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: intelgpu

// An attachment that cannot be satisfied is not particular to the close
// modifier. All it takes is a pointee whose corresponding storage is device
// memory while the pointer's corresponding storage is the original storage, and
// omp_target_associate_ptr gives the pointee such storage without any map
// modifier being involved.
//
// Three levels, p1 -> p2 -> x:
//
//  - x is associated with device memory, so it is device-bound, and nothing can
//    change that: the association is the program's own and its address has been
//    handed out;
//  - p1 is mapped by a construct of its own, so by the time anything is attached
//    to it, it is already present with its storage shared with the original;
//  - the last construct maps p2 afresh and prescribes both attachments.
//
// p2 must hold a device address, because x is device-bound and cannot be
// changed. That makes p2's own storage device memory, so p1 must hold a device
// address too -- but p1 is already present sharing storage with the original, so
// it cannot. Neither side of that attachment can move, and assigning a device
// address to the original p1 is not an option, so the runtime reports it.
//
#include <omp.h>
#include <stdio.h>

#pragma omp requires unified_shared_memory

int x[10];
int *p2;
int **p1;

int main(void) {
  // The unified_shared_memory requirement is registered when a device image is
  // loaded, so the program needs a target region for it to take effect.
#pragma omp target
  {
  }

  int dev = omp_get_default_device();

  int *x_device = (int *)omp_target_alloc(sizeof(int) * 10, dev);
  if (!x_device) {
    fprintf(stderr, "omp_target_alloc failed\n");
    return 1;
  }
  if (omp_target_associate_ptr(&x[0], x_device, sizeof(int) * 10, 0, dev)) {
    fprintf(stderr, "omp_target_associate_ptr failed\n");
    return 1;
  }

  // CHECK: x is device-bound
  fprintf(stderr, "x is %s\n",
          omp_get_mapped_ptr(&x[0], dev) == (void *)&x[0] ? "host-bound"
                                                         : "device-bound");

  p2 = &x[0];
  p1 = &p2;

  // Mapped on its own construct, so it is already present below.
#pragma omp target enter data map(alloc : p1)

  // CHECK: p1 is host-bound
  fprintf(stderr, "p1 is %s\n",
          omp_get_mapped_ptr(&p1, dev) == (void *)&p1 ? "host-bound"
                                                     : "device-bound");

  // p2 is new here, and both attachments are prescribed. No close modifier.
  //
  // clang-format off
  // CHECK: could not do pointer attachment
  // CHECK-SAME: would have to be device-bound as well
  // clang-format on
#pragma omp target enter data map(alloc : p2) map(alloc : p2[0 : 10])          \
    map(alloc : p1[0 : 1])

  fprintf(stderr, "unreachable\n");
  return 0;
}
