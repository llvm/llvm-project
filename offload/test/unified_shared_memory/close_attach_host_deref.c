// RUN: %libomptarget-compile-generic
// RUN: %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefixes=CHECK,DEFAULT
//
// RUN: %libomptarget-compile-generic -DVIA_ALWAYS=1
// RUN: env LIBOMPTARGET_TREAT_ATTACH_AUTO_AS_ALWAYS=1 \
// RUN: %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefixes=CHECK,ALWAYS

// REQUIRES: unified_shared_memory
// UNSUPPORTED: clang-6, clang-7, clang-8, clang-9

// amdgpu runtime crash
// Fails on nvptx with error: an illegal memory access was encountered
// UNSUPPORTED: amdgcn-amd-amdhsa
// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: intelgpu

// Dereferencing a pointer on the host while its corresponding pointer is in the
// attached state, under unified shared memory.
//
// The pointer is mapped without close, so its corresponding storage is the
// original storage. The pointee is mapped with close, so it gets a device
// buffer of its own, and pointer attachment then writes that device address
// into the corresponding pointer -- which is the original pointer. A host
// dereference afterwards goes through the device address.
//
// Two orderings are covered. In the default configuration the pointee is mapped
// with close after the pointer, so the attachment is triggered by the pointee
// becoming newly mapped. With VIA_ALWAYS the pointee already has device storage
// and the attachment is triggered separately, by a map of the zero-length array
// section under LIBOMPTARGET_TREAT_ATTACH_AUTO_AS_ALWAYS (OpenMP 6.0 has no
// attach map-type-modifier for C/C++, so the environment variable stands in for
// attach(always)).
//
// FIXME: the value checked below is the one produced today; the expected value
// is given alongside it. Reading x[0] through p is the natural thing for a
// program to do here, and it does not modify the pointer, so nothing in the
// OpenMP 6.0 map clause restrictions forbids it.

#include <stdio.h>

#pragma omp requires unified_shared_memory

int x[10];
int *p = &x[0];

int main() {
  for (int i = 0; i < 10; ++i)
    x[i] = 42;

  // The unified_shared_memory requirement is registered when a device image is
  // loaded, so the program needs a target region for it to take effect.
#pragma omp target
  {
  }

  // CHECK: before: p == &x[0]
  printf("before: p %s &x[0]\n", p == &x[0] ? "==" : "!=");

#if VIA_ALWAYS
  // Pointer and pointee both get their storage first, then attachment is
  // triggered on its own.
#pragma omp target enter data map(alloc : p)
#pragma omp target enter data map(close, alloc : x[0 : 10])
#pragma omp target enter data map(alloc : p[0 : 0])
#else
  // The pointee is newly mapped with close, which triggers the attachment.
#pragma omp target enter data map(alloc : p)
#pragma omp target enter data map(close, alloc : p[0 : 10])
#endif

#if VIA_ALWAYS
  // EXPECTED: after attach: p == &x[0]
  // ALWAYS:   after attach: p != &x[0]
  // FIXME: here the pointer and the pointee were both already present when the
  // attachment was prescribed, so neither could be given a different backing at
  // creation time, and the device address was written into the original p. This
  // case needs a diagnostic rather than a silent choice.
#else
  // The pointee was kept on the host path, because attaching the pointer would
  // otherwise have written a device address into the original p.
  // DEFAULT: after attach: p == &x[0]
#endif
  printf("after attach: p %s &x[0]\n", p == &x[0] ? "==" : "!=");

  // CHECK: Done!
  printf("Done!\n");
  return 0;
}
