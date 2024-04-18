// clang-format off
// RUN: %libomptarget-compilexx-generic
// RUN: env OMPX_EAGER_ZERO_COPY_MAPS=1 HSA_XNACK=1 LIBOMPTARGET_INFO=128 %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefix=INFO_ZERO_EAGER -check-prefix=INFO

// RUN: %libomptarget-compilexx-generic -DUSE_USM=1
// RUN: env OMPX_EAGER_ZERO_COPY_MAPS=1 HSA_XNACK=1 LIBOMPTARGET_INFO=128 %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefix=INFO_USM_EAGER -check-prefix=INFO

// RUN: %libomptarget-compilexx-generic
// RUN: env OMPX_EAGER_ZERO_COPY_MAPS=1 HSA_XNACK=0 LIBOMPTARGET_INFO=128 %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefix=INFO_ZERO_EAGER_NO_XNACK

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO

// REQUIRES: unified_shared_memory
// REQUIRES: apu

// clang-format on

#ifdef USE_USM
#pragma omp requires unified_shared_memory
#endif

int main() {
  int a = -1;
  // clang-format off
  // INFO: XNACK is enabled.
  // INFO_ZERO_EAGER: Application configured to run in zero-copy using auto zero-copy.
  // INFO_USM_EAGER: Application configured to run in zero-copy using unified_shared_memory.
  // INFO: Requested pre-faulting of GPU page tables.

  // INFO_ZERO_EAGER_NO_XNACK: Application configured to run in zero-copy using auto zero-copy.
  // INFO_ZERO_EAGER_NO_XNACK: Requested pre-faulting of GPU page tables.
  // clang-format on
#pragma omp target map(tofrom : a)
  { a++; }
  return 0;
}
