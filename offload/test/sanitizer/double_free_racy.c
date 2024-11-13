// clang-format off
// RUN: %libomptarget-compileopt-generic
// RUN: %not --crash env -u LLVM_DISABLE_SYMBOLIZATION OFFLOAD_TRACK_ALLOCATION_TRACES=1 %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK
// RUN: %libomptarget-compileopt-generic -g
// RUN: %not --crash env -u LLVM_DISABLE_SYMBOLIZATION OFFLOAD_TRACK_ALLOCATION_TRACES=1 %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK
// clang-format on

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu-LTO
// UNSUPPORTED: s390x-ibm-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu-LTO

#include <omp.h>

int main(void) {
  void *Ptr1 = omp_target_alloc(8, 0);
#pragma omp parallel num_threads(4)
  omp_target_free(Ptr1, 0);
}

// CHECK: OFFLOAD ERROR: double-free of device memory: 0x
// CHECK   dataDelete
// CHECK:  omp_target_free
//
// CHECK: Last deallocation:
// CHECK:  dataDelete
// CHECK:  omp_target_free

// CHECK: Last allocation of size 8 -> device pointer
// CHECK:  dataAlloc
// CHECK:  omp_target_alloc
