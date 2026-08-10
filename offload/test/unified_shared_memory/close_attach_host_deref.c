// RUN: %libomptarget-compile-generic
// RUN: %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefixes=CHECK,DEFAULT
//
// RUN: %libomptarget-compile-generic -DVIA_ALWAYS=1
// RUN: env LIBOMPTARGET_TREAT_ATTACH_AUTO_AS_ALWAYS=1 \
// RUN: %libomptarget-run-fail-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefixes=ALWAYS

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
// Two orderings are covered, and they differ in whether the situation can be
// resolved at all.
//
// In the default configuration the pointee is mapped with close after the
// pointer, so its allocation was made by that construct and can be given up:
// the pointee returns to sharing storage with the original, the attached address
// is the original one, and the original p is left alone.
//
// With VIA_ALWAYS the pointee already has device storage and the attachment is
// triggered separately, by a map of the zero-length array section under
// LIBOMPTARGET_TREAT_ATTACH_AUTO_AS_ALWAYS (OpenMP 6.0 has no attach
// map-type-modifier for C/C++, so the environment variable stands in for
// attach(always)). Both sides were already present, so neither can change its
// storage now: the pointee's allocation may already have been handed to the
// program, and so may p's device address. There is no way to attach without
// assigning a device address to the original p, so the runtime reports it
// instead of doing so silently.

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
  //
  // clang-format off
  // ALWAYS: could not do pointer attachment
  // ALWAYS-SAME: host-bound mapping and the pointee is device-bound
  // clang-format on
#pragma omp target enter data map(alloc : p[0 : 0])
#else
  // The pointee is newly mapped with close, which triggers the attachment.
#pragma omp target enter data map(alloc : p)
#pragma omp target enter data map(close, alloc : p[0 : 10])
#endif

  // The pointee gave its allocation up, because attaching the pointer would
  // otherwise have written a device address into the original p.
  // DEFAULT: after attach: p == &x[0]
  printf("after attach: p %s &x[0]\n", p == &x[0] ? "==" : "!=");

  // DEFAULT: Done!
  printf("Done!\n");
  return 0;
}
