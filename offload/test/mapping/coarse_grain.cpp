// clang-format off
// RUN: %libomptarget-compilexx-generic
// RUN: env HSA_XNACK=1 LIBOMPTARGET_INFO=30 %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefix=CHECK

// RUN: %libomptarget-compilexx-generic
// RUN: env OMPX_DISABLE_USM_MAPS=1 HSA_XNACK=1 LIBOMPTARGET_INFO=30 %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefix=CHECK_FINE

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu-LTO

// REQUIRES: unified_shared_memory
// REQUIRES: mi200

// clang-format on

#include <cstdio>
#include <omp.h>

#pragma omp requires unified_shared_memory

int main() {
  const size_t n = 1024;

  double *a = new double[n];
  // clang-format off
  // CHECK: Memory pages for HstPtrBegin 0x{{.*}} Size=8192 switched to coarse grain
  // CHECK: Before mapping, memory is fine grain.
  // CHECK_FINE: Before mapping, memory is fine grain.
  // clang-format on
  if (omp_is_coarse_grain_mem_region(a, n * sizeof(double)))
    printf("Before mapping, memory is coarse grain.\n");
  else
    printf("Before mapping, memory is fine grain.\n");

#pragma omp target enter data map(to : a[:n])

  // CHECK: After mapping, memory is coarse grain.
  // CHECK_FINE: After mapping, memory is fine grain.
  if (omp_is_coarse_grain_mem_region(a, n * sizeof(double)))
    printf("After mapping, memory is coarse grain.\n");
  else
    printf("After mapping, memory is fine grain.\n");

#pragma omp target exit data map(from : a[:n])

  // CHECK: After removing map, memory is still coarse grain.
  // CHECK_FINE: After removing map, memory is back to fine grain.
  if (omp_is_coarse_grain_mem_region(a, n * sizeof(double)))
    printf("After removing map, memory is still coarse grain.\n");
  else
    printf("After removing map, memory is back to fine grain.\n");

// Plugins must be initialized for unified_shared_memory requirement
// to be added. An empty target region is enough for that initialization.
#pragma omp target
  {}

  return 0;
}
