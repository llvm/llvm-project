// RUN: %libomptarget-compile-run-and-check-generic

// REQUIRES: unified_shared_memory
// UNSUPPORTED: clang-6, clang-7, clang-8, clang-9

// amdgpu runtime crash
// Fails on nvptx with error: an illegal memory access was encountered
// UNSUPPORTED: amdgcn-amd-amdhsa
// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: intelgpu

// Which side of an attachment changes has to be decided for the attachments
// that have no choice before those that do.
//
// A pointee whose allocation was made by an enclosing construct cannot give it
// up, so such an attachment can only be settled by giving the pointer device
// storage. A pointee this construct allocated can be released instead, which is
// preferred because it settles that attachment outright.
//
// Here both pointers live in the same structure, so the forced decision for one
// of them determines the storage of the other's pointer too:
//
//   s.p1 -> a1   a1 allocated by this construct, so it could be released
//   s.p2 -> a2   a2 allocated earlier, so s must hold a device address
//
// Deciding the releases first would release a1 while s was still the original
// storage, and the s.p2 attachment would then give s device storage anyway --
// leaving a1 released for nothing and its close request unmet. Settling the
// forced attachment first leaves nothing for the release pass to do.

#include <omp.h>
#include <stdio.h>

#pragma omp requires unified_shared_memory

int a1[10], a2[10];

struct S {
  int *p1;
  int *p2;
};

struct S s;

int main(void) {
  for (int i = 0; i < 10; ++i) {
    a1[i] = 1;
    a2[i] = 2;
  }
  s.p1 = &a1[0];
  s.p2 = &a2[0];

  // The unified_shared_memory requirement is registered when a device image is
  // loaded, so the program needs a target region for it to take effect.
#pragma omp target
  {
  }

  int dev = omp_get_default_device();

  // a2 is allocated here, so it cannot be released by the construct below.
#pragma omp target enter data map(close, to : a2[0 : 10])

#pragma omp target enter data map(alloc : s) map(close, to : s.p1[0 : 10])     \
    map(alloc : s.p2[0 : 0])

  // s holds device addresses, so a1 never needed releasing and its close
  // request is met.
  // CHECK: s is device-bound
  printf("s is %s\n", omp_get_mapped_ptr(&s, dev) == (void *)&s ? "host-bound"
                                                                : "device-bound");

  // CHECK: a1 is device-bound
  printf("a1 is %s\n", omp_get_mapped_ptr(&a1[0], dev) == (void *)&a1[0]
                           ? "host-bound"
                           : "device-bound");

  // CHECK: a2 is device-bound
  printf("a2 is %s\n", omp_get_mapped_ptr(&a2[0], dev) == (void *)&a2[0]
                           ? "host-bound"
                           : "device-bound");

  // The original pointers are intact either way.
  // CHECK: host: s.p1 == &a1[0], s.p2 == &a2[0]
  printf("host: s.p1 %s &a1[0], s.p2 %s &a2[0]\n",
         s.p1 == &a1[0] ? "==" : "!=", s.p2 == &a2[0] ? "==" : "!=");

  // CHECK: Done!
  printf("Done!\n");
  return 0;
}
