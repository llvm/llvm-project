// RUN: %libomptarget-compile-run-and-check-generic

// REQUIRES: unified_shared_memory
// UNSUPPORTED: clang-6, clang-7, clang-8, clang-9

// amdgpu runtime crash
// Fails on nvptx with error: an illegal memory access was encountered
// UNSUPPORTED: amdgcn-amd-amdhsa
// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: intelgpu

// An allocation made for a close mapping under unified shared memory may still
// be released, if a pointer that has to keep the original storage is attached to
// it. The initial transfer into such an allocation is therefore held back until
// the storage is settled, and either issued or dropped.
//
// Both outcomes have to leave the right data visible, which is what this checks.
// Where the allocation is released the device reads the original storage, which
// already holds the values; where it is kept the transfer has to have happened.

#include <stdio.h>

#pragma omp requires unified_shared_memory

int released[10];
int kept[10];
int *p_released = &released[0];
int *p_kept = &kept[0];

int sum_released, sum_kept;

int main(void) {
  for (int i = 0; i < 10; ++i) {
    released[i] = i + 1;
    kept[i] = i + 1;
  }

  // The unified_shared_memory requirement is registered when a device image is
  // loaded, so the program needs a target region for it to take effect.
#pragma omp target
  {
  }

  // The pointer is mapped afresh alongside it, so the allocation for this close
  // mapping is released and its transfer dropped.
#pragma omp target enter data map(alloc : p_released)                          \
    map(close, to : p_released[0 : 10])

  // Here the pointee is allocated by its own construct, so the pointer is given
  // device storage instead and the transfer is issued.
#pragma omp target enter data map(close, to : kept[0 : 10])
#pragma omp target enter data map(alloc : p_kept) map(alloc : p_kept[0 : 0])

#pragma omp target map(present, alloc : p_released, p_kept)                    \
    map(from : sum_released, sum_kept)
  {
    sum_released = 0;
    sum_kept = 0;
    for (int i = 0; i < 10; ++i) {
      sum_released += p_released[i];
      sum_kept += p_kept[i];
    }
  }

  // CHECK: released: 55, kept: 55
  printf("released: %d, kept: %d\n", sum_released, sum_kept);

  // CHECK: host: p_released == &released[0], p_kept == &kept[0]
  printf("host: p_released %s &released[0], p_kept %s &kept[0]\n",
         p_released == &released[0] ? "==" : "!=",
         p_kept == &kept[0] ? "==" : "!=");

  // CHECK: Done!
  printf("Done!\n");
  return 0;
}
