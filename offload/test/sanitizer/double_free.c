// clang-format off
// RUN: %libomptarget-compileopt-generic
// RUN: %not --crash env LIBOMPTARGET_MEMORY_MANAGER_THRESHOLD=0 %libomptarget-run-generic 2>&1 | %fcheck-generic
// clang-format on

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu

// Pooled memory is only returned to the pool, so a second free would go
// unnoticed. Disable the memory manager to make the free real.

#include <omp.h>

int main(void) {
  void *Ptr1 = omp_target_alloc(8, 0);
  omp_target_free(Ptr1, 0);
  void *Ptr2 = omp_target_alloc(8, 0);
  omp_target_free(Ptr2, 0);
  void *Ptr3 = omp_target_alloc(8, 0);
  omp_target_free(Ptr3, 0);
  omp_target_free(Ptr2, 0);
}

// The runtime rejects a pointer that is no longer a live allocation.
// CHECK: Failure to deallocate device pointer 0x{{.*}}: pointer is not a known allocation in this context
// CHECK: Failed to deallocate device ptr
