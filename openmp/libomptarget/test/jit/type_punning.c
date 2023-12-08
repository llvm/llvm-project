// clang-format off
//
// RUN: %libomptarget-compileopt-generic -fopenmp-target-jit
// RUN: env LIBOMPTARGET_JIT_PRE_OPT_IR_MODULE=%t.pre.ll     \
// RUN:     LIBOMPTARGET_JIT_SKIP_OPT=true                   \
// RUN:     %libomptarget-run-generic
// RUN: %fcheck-plain-generic --input-file %t.pre.ll %s
//
// clang-format on

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO

// Ensure that there is only the kernel function left, not any outlined
// parallel regions.
//
// CHECK: define
// CHECK-NOT: define

#include <omp.h>
#include <stdio.h>

void f(long *A, int N) {
  long i = 0;
#pragma omp target map(A[ : N])
  {
#pragma omp parallel firstprivate(i)
    A[omp_get_thread_num()] = i;
#pragma omp parallel firstprivate(i, N)
    A[omp_get_thread_num()] += i + N;
  }
}

int main() {
  long A[1];
  f(&A[0], 1);
  printf("%li\n", A[0]);
  return 0;
}
