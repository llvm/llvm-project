// clang-format off
// RUN: %libomptarget-compileopt-generic
// RUN: %not --crash env -u LLVM_DISABLE_SYMBOLIZATION OFFLOAD_TRACK_ALLOCATION_TRACES=1 %libomptarget-run-generic 2>&1 | %fcheck-generic
// clang-format on

// The allocation of a map is reported with the host pointer it was issued for,
// which only the OpenMP runtime, not the plugin, knows about.

// REQUIRES: gpu
//
// Maps are not backed by their own allocation when zero-copy is used.
// UNSUPPORTED: apu

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
  int Device = omp_get_default_device();
  int *A = (int *)malloc(10 * sizeof(int));

#pragma omp target enter data map(to : A[0 : 10])
#pragma omp target
  {
    ;
  }

  void *DevicePtr = omp_get_mapped_ptr(A, Device);
  fprintf(stderr, "mapped %p\n", (void *)A);

  /// Free the allocation of the mapping behind the runtime's back, so that the
  /// second deallocation is a double-free of an allocation that was issued for
  /// a host pointer.
  omp_target_free(DevicePtr, Device);
  omp_target_free(DevicePtr, Device);

  return 0;
}

// clang-format off
// CHECK: mapped 0x{{0*}}[[HST_PTR:[0-9a-f]+]]
// CHECK: OFFLOAD ERROR: double-free of device memory: 0x{{0*}}[[DEV_PTR:[0-9a-f]+]]
// CHECK: Last deallocation:
// CHECK: Last allocation of size 40 for host pointer 0x{{0*}}[[HST_PTR]] -> device pointer 0x{{0*}}[[DEV_PTR]]:
// clang-format on
