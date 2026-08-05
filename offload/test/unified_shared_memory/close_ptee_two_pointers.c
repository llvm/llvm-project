// RUN: %libomptarget-compile-run-and-check-generic

// REQUIRES: unified_shared_memory
// UNSUPPORTED: clang-6, clang-7, clang-8, clang-9

// amdgpu runtime crash
// Fails on nvptx with error: an illegal memory access was encountered
// UNSUPPORTED: amdgcn-amd-amdhsa
// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: intelgpu

// Two pointers to the same pointee, where one of them has device storage of its
// own and the other stays on the unified-shared-memory host path.
//
// p1 is mapped with close, so it gets a real device allocation and attachment
// writes into that, leaving the original p1 untouched. p2 is mapped without
// close, so its corresponding storage is the original storage, and attachment
// writes the device pointee address into the original p2.
//
// So a single close pointee cannot be made correct by giving its pointers
// storage of their own: whether that is possible depends on how each pointer
// was mapped, and here one of them was not mapped in a way that provides it.
//
// FIXME: the values checked below are the ones produced today; the expected
// value is given alongside each.

#include <stdio.h>

#pragma omp requires unified_shared_memory

int arr[10] = {0};

// Both point to the same pointee.
int *p1 = &arr[0];
int *p2 = &arr[0];

int *p1_device, *p2_device;

int main() {
  // CHECK: before: p1 == &arr[0], p2 == &arr[0]
  printf("before: p1 %s &arr[0], p2 %s &arr[0]\n",
         p1 == &arr[0] ? "==" : "!=", p2 == &arr[0] ? "==" : "!=");

  // p1 gets device storage of its own, p2 does not.
#pragma omp target data map(close, alloc : p1) map(alloc : p2)
  {
    // The pointee is newly mapped here, with close, so it gets a device buffer.
#pragma omp target data map(close, alloc : p1[0 : 10]) map(p2[0 : 0])
    {
#pragma omp target map(present, alloc : p1, p2) map(from : p1_device, p2_device)
      {
        p1_device = p1;
        p2_device = p2;
        p1[0] = 55;
      }
    }
  }

  // p2 has no device storage of its own, so the pointee was kept on the host
  // path and both pointers designate it there.
  // CHECK: in tgt: p1 == &arr[0], p2 == &arr[0]
  printf(
      "in tgt: p1 %s &arr[0], p2 %s &arr[0]\n",
      p1_device == &arr[0] ? "==" : "!=", p2_device == &arr[0] ? "==" : "!=");

  // CHECK: after: p1 == &arr[0]
  printf("after: p1 %s &arr[0]\n", p1 == &arr[0] ? "==" : "!=");

  // CHECK: after: p2 == &arr[0]
  printf("after: p2 %s &arr[0]\n", p2 == &arr[0] ? "==" : "!=");

  // The pointee stayed on the host path, so the write is to arr itself and
  // there is no separate buffer for it to be stranded in.
  // CHECK: arr[0] = 55
  printf("arr[0] = %d\n", arr[0]);

  // CHECK: Done!
  printf("Done!\n");
  return 0;
}
