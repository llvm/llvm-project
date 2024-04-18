// clang-format off
// RUN: %libomptarget-compilexx-generic
// RUN: env OMPX_APU_MAPS=1 HSA_XNACK=1 LIBOMPTARGET_INFO=128 %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefix=INFO_ZERO -check-prefix=INFO -check-prefix=CHECK

// RUN: %libomptarget-compilexx-generic -DUSE_USM=1
// RUN: env OMPX_APU_MAPS=1 HSA_XNACK=1 LIBOMPTARGET_INFO=128 %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefix=INFO_USM -check-prefix=INFO -check-prefix=CHECK

// RUN: %libomptarget-compilexx-generic -DUSE_USM=1 -DNO_ACCESS=1
// RUN: env HSA_XNACK=0 LIBOMPTARGET_INFO=128 %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefix=INFO_ERR -check-prefix=CHECK


// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO

// REQUIRES: unified_shared_memory

// clang-format on

#include <cstdio>

#if USE_USM == 1
#pragma omp requires unified_shared_memory
#endif

int main() {
  int a = -1;
  // clang-format off
  // INFO: XNACK is enabled.
  // INFO_ZERO: Application configured to run in zero-copy using auto zero-copy.
  // INFO_USM: Application configured to run in zero-copy using unified_shared_memory.

  // INFO_ERR: XNACK is disabled.
  // INFO_ERR: Application configured to run in zero-copy using unified_shared_memory.
  // INFO_ERR: Running a program that requires XNACK on a system where XNACK is disabled. This may cause problems when using an OS-allocated pointer inside a target region. Re-run with HSA_XNACK=1 to remove this warning.

  // clang-format on
  int x = 0;
#pragma omp target map(tofrom : a)
  {
#if NO_ACCESS != 1
    a++;
#endif
    x++;
  }
  // CHECK: PASS
  printf("PASS\n");
  return 0;
}
