// RUN: %libomptarget-compile-run-and-check-generic

// REQUIRES: unified_shared_memory
// UNSUPPORTED: clang-6, clang-7, clang-8, clang-9

// amdgpu runtime crash
// Fails on nvptx with error: an illegal memory access was encountered
// UNSUPPORTED: amdgcn-amd-amdhsa
// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: intelgpu

// A chain of two pointers, where the second one lives inside what the first one
// points to: p1 refers to a structure whose member p2 refers to an array.
//
// p1 is attached to the structure first. Giving the structure device storage
// afterwards -- because a close mapping of what p2 refers to needs p2 to have a
// corresponding pointer distinct from the original -- moves the structure's
// device address, so the value already attached to p1 no longer designates it.
// Nothing revisits p1, so the device copy of p1 keeps pointing at the original
// structure, and device code reaching p2 through p1 does not see the close
// buffer.
//
// Any mechanism that gives an entry device storage after a pointer has been
// attached to it therefore has to revisit the pointers already attached to that
// entry, transitively.
//
// FIXME: the values checked below are the ones produced today; the expected
// value is given alongside each.

#include <stdio.h>

#pragma omp requires unified_shared_memory

struct Inner {
  int *p2;
  int pad;
};

int leaf[10];
struct Inner inner;
struct Inner *p1 = &inner;

struct Inner *p1_device;
int *p2_device;

int main() {
  for (int i = 0; i < 10; ++i)
    leaf[i] = 42;
  inner.p2 = &leaf[0];

  // The unified_shared_memory requirement is registered when a device image is
  // loaded, so the program needs a target region for it to take effect.
#pragma omp target
  {
  }

  // p1 is attached to the structure.
#pragma omp target enter data map(alloc : p1)
#pragma omp target enter data map(alloc : p1[0 : 1])

  // The close mapping needs p2, which lives inside the structure, to have a
  // corresponding pointer distinct from the original.
#pragma omp target enter data map(close, alloc : p1->p2[0 : 10])

#pragma omp target map(present, alloc : p1, p1[0 : 1])                         \
    map(from : p1_device, p2_device)
  {
    p1_device = p1;
    p2_device = p1->p2;
  }

  // The structure was kept on the host path, so p1 designates it there and the
  // original p1 is unchanged.
  // CHECK: device p1 == &inner
  printf("device p1 %s &inner\n", p1_device == &inner ? "==" : "!=");

  // CHECK: device p1->p2 == &leaf[0]
  printf("device p1->p2 %s &leaf[0]\n", p2_device == &leaf[0] ? "==" : "!=");

  // CHECK: host: p1 == &inner, inner.p2 == &leaf[0]
  printf("host: p1 %s &inner, inner.p2 %s &leaf[0]\n",
         p1 == &inner ? "==" : "!=", inner.p2 == &leaf[0] ? "==" : "!=");

  // CHECK: Done!
  printf("Done!\n");
  return 0;
}
