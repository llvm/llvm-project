// clang-format off
// RUN: %libomptarget-compileoptxx-generic
// RUN: %not --crash env -u LLVM_DISABLE_SYMBOLIZATION OFFLOAD_TRACK_ALLOCATION_TRACES=1 %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK,NDEBG
// RUN: %libomptarget-compileoptxx-generic -g
// RUN: %not --crash env -u LLVM_DISABLE_SYMBOLIZATION OFFLOAD_TRACK_ALLOCATION_TRACES=1 %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK,DEBUG
// clang-format on

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu-LTO
// UNSUPPORTED: s390x-ibm-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu-LTO

#include <omp.h>

extern "C" {
void *llvm_omp_target_alloc_shared(size_t Size, int DeviceNum);
void llvm_omp_target_free_host(void *Ptr, int DeviceNum);
}

int main(void) {
  void *P = llvm_omp_target_alloc_shared(8, 0);
  llvm_omp_target_free_host(P, 0);
}

// clang-format off
// CHECK: OFFLOAD ERROR: deallocation requires pinned host memory but allocation was managed memory: 0x
// CHECK:  dataDelete
// CHECK:  llvm_omp_target_free_host
// NDEBG: main
// DEBUG:  main {{.*}}free_wrong_ptr_kind.cpp:25
//
// CHECK: Last allocation of size 8 -> device pointer
// CHECK:  dataAlloc
// CHECK:  llvm_omp_target_alloc_shared
// NDEBG:  main
// DEBUG:  main {{.*}}free_wrong_ptr_kind.cpp:24
