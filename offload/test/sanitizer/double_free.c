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

int main(void) {
  void *Ptr1 = omp_target_alloc(8, 0);
  omp_target_free(Ptr1, 0);
  void *Ptr2 = omp_target_alloc(8, 0);
  omp_target_free(Ptr2, 0);
  void *Ptr3 = omp_target_alloc(8, 0);
  omp_target_free(Ptr3, 0);
  omp_target_free(Ptr2, 0);
}

// CHECK: OFFLOAD ERROR: double-free of device memory: 0x
// CHECK:   dataDelete
// CHECK:   omp_target_free
// NDEBG:   main
// DEBUG:   main {{.*}}double_free.c:25
//
// CHECK: Last deallocation:
// CHECK:  dataDelete
// CHECK:  omp_target_free
// NDEBG:  main
// DEBUG:  main {{.*}}double_free.c:24
//
// CHECK: Last allocation of size 8 -> device pointer
// CHECK:  dataAlloc
// CHECK:  omp_target_alloc
// NDEBG:  main
// DEBUG:  main {{.*}}double_free.c:23
//
// CHECK: Prior allocations with the same base pointer:
// CHECK: #0 Prior deallocation of size 8:
// CHECK:  dataDelete
// CHECK:  omp_target_free
// NDEBG:  main
// DEBUG:  main {{.*}}double_free.c:22
//
// CHECK: #0 Prior allocation -> device pointer
// CHECK:  dataAlloc
// CHECK:  omp_target_alloc
// NDEBG:  main
// DEBUG:  main {{.*}}double_free.c:20
//
// CHECK: #1 Prior deallocation of size 8:
// CHECK:  dataDelete
// CHECK:  omp_target_free
// NDEBG:  main
// DEBUG:  main {{.*}}double_free.c:20
//
// CHECK: #1 Prior allocation -> device pointer
// CHECK:  dataAlloc
// CHECK:  omp_target_alloc
// NDEBG:  main
// DEBUG:  main {{.*}}double_free.c:19
