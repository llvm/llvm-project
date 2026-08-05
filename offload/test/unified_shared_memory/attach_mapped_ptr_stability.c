// RUN: %libomptarget-compile-run-and-check-generic

// REQUIRES: unified_shared_memory
// UNSUPPORTED: clang-6, clang-7, clang-8, clang-9

// amdgpu runtime crash
// Fails on nvptx with error: an illegal memory access was encountered
// UNSUPPORTED: amdgcn-amd-amdhsa
// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: intelgpu

// The device address of a mapped list item must not change while it is mapped.
//
// A program can obtain it with omp_get_mapped_ptr, and can also hold it from
// use_device_ptr or use_device_addr, or have passed it to a device already. So
// whether an entry's storage is shared with the original can only be decided
// when the entry is created: giving an already-present entry a device
// allocation later would invalidate every device address obtained for it
// beforehand.
//
// This matters for pointer attachment under unified shared memory, where a
// pointer whose storage is shared with the original cannot be attached without
// writing a device address into the original pointer. Resolving that by giving
// the pointer storage after the fact is what this test rules out.

#include <omp.h>
#include <stdio.h>

#pragma omp requires unified_shared_memory

int arr[10];
int *p = &arr[0];

int main() {
  int dev = omp_get_default_device();

  // The unified_shared_memory requirement is registered when a device image is
  // loaded, so the program needs a target region for it to take effect.
#pragma omp target
  {
  }

#pragma omp target enter data map(alloc : p)

  void *Before = omp_get_mapped_ptr(&p, dev);

  // Attaching p to a pointee that has its own device storage must not change
  // the device address of p itself. Here the pointee is newly mapped with
  // close, which is what prescribes the attachment.
#pragma omp target enter data map(close, alloc : p[0 : 10])

  void *After = omp_get_mapped_ptr(&p, dev);

  // CHECK: device address of p is stable: yes
  printf("device address of p is stable: %s\n", Before == After ? "yes" : "no");

  // CHECK: Done!
  printf("Done!\n");
  return 0;
}
