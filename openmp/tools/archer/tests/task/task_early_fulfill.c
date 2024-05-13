// RUN: %libarcher-compile -fopenmp-version=50 && env OMP_NUM_THREADS='3' \
// RUN:    %libarcher-run
//| FileCheck %s

// Checked gcc 10.1 still does not support detach clause on task construct.
// UNSUPPORTED: gcc-4, gcc-5, gcc-6, gcc-7, gcc-8, gcc-9, gcc-10
// gcc 11 introduced detach clause, but gomp interface in libomp has no support
// XFAIL: gcc-11, gcc-12
// clang supports detach clause since version 11.
// UNSUPPORTED: clang-10, clang-9, clang-8, clang-7
// icc compiler does not support detach clause.
// UNSUPPORTED: icc
// REQUIRES: tsan

#include <omp.h>
#include <stdio.h>

int main() {
#pragma omp parallel
#pragma omp master
  {
    omp_event_handle_t event;
#pragma omp task detach(event) if (0)
    { omp_fulfill_event(event); }
#pragma omp taskwait
  }
  return 0;
}
