// RUN: %libomptarget-compile-run-and-check-generic

// REQUIRES: unified_shared_memory
// UNSUPPORTED: clang-6, clang-7, clang-8, clang-9

// amdgpu runtime crash
// Fails on nvptx with error: an illegal memory access was encountered
// UNSUPPORTED: amdgcn-amd-amdhsa
// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: intelgpu

// The pointee already has device storage before the pointers are mapped, so no
// allocation happens for it on the inner construct and there is nothing for the
// pointee to give up: it is simply already on the device.
//
// p2 is mapped without close, so its corresponding storage would be the original
// storage, and attaching it would write the device pointee address into the
// original p2. The host could then observe that value for as long as p2 remained
// attached, which under OpenMP 6.0 lasts until the pointer's storage is removed
// from the device data environment -- there is no detachment.
//
// This case is therefore not resolved from the pointee's side. It is resolved
// from the pointer's side: p2's own mapping is created by this construct, so p2
// can be given a device allocation of its own, and the assignment then reaches
// that instead of the original p2.
//
// Note what OpenMP 6.0 does and does not say here. Attachment assigns the
// corresponding pointer (7.9.6), and the corresponding storage may share
// storage with the original (7.9.6, 1.3.2), in which case the assignment would
// be observable on the host. Nothing preserves the original value during that
// window, and nothing restores it: the map-exiting sequence has no detach step.
// The one place the specification confronts the same situation, for self maps,
// requires runtime error termination when "the list item is a pointer that
// would be assigned a different value as a result of pointer attachment"
// (7.9.6), which indicates the intent is for this configuration not to arise --
// so the implementation has to keep it from arising.

#include <stdio.h>

#pragma omp requires unified_shared_memory

int arr[10] = {0};

int *p1 = &arr[0];
int *p2 = &arr[0];

int *p1_device, *p2_device;

int main() {
  // The pointee gets device storage before either pointer is mapped.
#pragma omp target enter data map(close, alloc : arr[0 : 10])

  // CHECK: before: p1 == &arr[0], p2 == &arr[0]
  printf("before: p1 %s &arr[0], p2 %s &arr[0]\n",
         p1 == &arr[0] ? "==" : "!=", p2 == &arr[0] ? "==" : "!=");

#pragma omp target data map(close, alloc : p1, p1[0 : 0])                      \
    map(alloc : p2, p2[0 : 0])
  {
#pragma omp target map(present, alloc : p1, p2) map(from : p1_device, p2_device)
    {
      p1_device = p1;
      p2_device = p2;
    }

    // Read on the host while both pointers are still attached.
    //
    // p1 has device storage of its own, so its original is unaffected.
    // CHECK: inside: p1 == &arr[0]
    printf("inside: p1 %s &arr[0]\n", p1 == &arr[0] ? "==" : "!=");

    // p2 was given device storage of its own, so its original is unaffected too.
    // CHECK: inside: p2 == &arr[0]
    printf("inside: p2 %s &arr[0]\n", p2 == &arr[0] ? "==" : "!=");
  }

  // CHECK: in tgt: p1 != &arr[0], p2 != &arr[0]
  printf(
      "in tgt: p1 %s &arr[0], p2 %s &arr[0]\n",
      p1_device == &arr[0] ? "==" : "!=", p2_device == &arr[0] ? "==" : "!=");

  // CHECK: after: p1 == &arr[0]
  printf("after: p1 %s &arr[0]\n", p1 == &arr[0] ? "==" : "!=");

  // CHECK: after: p2 == &arr[0]
  printf("after: p2 %s &arr[0]\n", p2 == &arr[0] ? "==" : "!=");

  // CHECK: Done!
  printf("Done!\n");
  return 0;
}
