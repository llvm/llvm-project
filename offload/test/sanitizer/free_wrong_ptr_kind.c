// clang-format off
// RUN: %libomptarget-compileopt-generic
// RUN: %not --crash env -u LLVM_DISABLE_SYMBOLIZATION OFFLOAD_TRACK_ALLOCATION_TRACES=1 %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK,NDEBG
// RUN: %libomptarget-compileopt-generic -g
// RUN: %not --crash env -u LLVM_DISABLE_SYMBOLIZATION OFFLOAD_TRACK_ALLOCATION_TRACES=1 %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK,DEBUG
// clang-format on

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu-LTO
// UNSUPPORTED: s390x-ibm-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu-LTO

#include <omp.h>

void *llvm_omp_target_alloc_host(size_t Size, int DeviceNum);

int main(void) {
  void *P = llvm_omp_target_alloc_host(8, 0);
  omp_target_free(P, 0);
}

// clang-format off
// CHECK: OFFLOAD ERROR: deallocation requires device memory but allocation was pinned host memory: 0x
// CHECK:  dataDelete
// CHECK:  omp_target_free
// NDEBG: main
// DEBUG:  main {{.*}}free_wrong_ptr_kind.c:22
//
// CHECK: Last allocation of size 8 -> device pointer
// CHECK:  dataAlloc
// CHECK:  llvm_omp_target_alloc_host
// NDEBG:  main
// DEBUG:  main {{.*}}free_wrong_ptr_kind.c:21
