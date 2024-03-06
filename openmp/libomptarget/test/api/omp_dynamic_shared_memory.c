// RUN: %libomptarget-compile-generic
// RUN: env LIBOMPTARGET_SHARED_MEMORY_SIZE=256 \
// RUN:   %libomptarget-run-generic | %fcheck-generic

// RUN: %libomptarget-compileopt-generic
// RUN: env LIBOMPTARGET_SHARED_MEMORY_SIZE=256 \
// RUN:   %libomptarget-run-generic | %fcheck-generic

// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO
// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: s390x-ibm-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu-LTO

#include <omp.h>
#include <stdio.h>

int main() {
  int x;
#pragma omp target parallel map(from : x)
  {
    int *buf = llvm_omp_target_dynamic_shared_alloc() + 252;
#pragma omp barrier
    if (omp_get_thread_num() == 0)
      *buf = 1;
#pragma omp barrier
    if (omp_get_thread_num() == 1)
      x = *buf;
  }

  // CHECK: PASS
  if (x == 1 && llvm_omp_target_dynamic_shared_alloc() == NULL)
    printf("PASS\n");
}
