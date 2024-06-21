// clang-format off
// RUN: %libomptarget-compileopt-generic
// RUN: env LIBOMPTARGET_INFO=16 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK,SPMD
// RUN: %libomptarget-compileopt-generic -mllvm --openmp-opt-disable-spmdization
// RUN: env LIBOMPTARGET_INFO=16 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK,GENERIC
// clang-format on

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO
// UNSUPPORTED: s390x-ibm-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu-LTO

#include <omp.h>
#include <stdio.h>

__attribute__((weak)) void noop() {}

int main(void) {
  int nthreads = 0, ip = 0, lvl = 0, alvl = 0, nested = 0, tid = 0, maxt = 0;

#pragma omp target map(from : nthreads, ip, lvl, alvl, nested, tid, maxt)
  {
    nthreads = omp_get_num_threads();
    ip = omp_in_parallel();
    lvl = omp_get_level();
    alvl = omp_get_active_level();
    nested = omp_get_nested();
    tid = omp_get_thread_num();
    maxt = omp_get_max_threads();
    #pragma omp parallel
    noop();
  }
  printf("NumThreads: %i, InParallel: %i, Level: %i, ActiveLevel: %i, Nested: %i, "
         "ThreadNum: %i, MaxThreads: %i\n",
         nthreads, ip, lvl, alvl, nested, tid, maxt);
  // GENERIC: Generic mode
  // SPMD: Generic-SPMD mode
  // CHECK: NumThreads: 1
  // CHECK: InParallel: 0
  // CHECK: Level: 0
  // CHECK: ActiveLevel: 0
  // CHECK: Nested: 0
  // CHECK: ThreadNum: 0
  // CHECK: MaxThreads:
  return 0;
}
