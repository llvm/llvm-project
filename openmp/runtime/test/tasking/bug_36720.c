// RUN: %libomp-compile-and-run

/*
Bugzilla: https://bugs.llvm.org/show_bug.cgi?id=36720

Assertion failure at kmp_runtime.cpp(1715): nthreads > 0.
OMP: Error #13: Assertion failure at kmp_runtime.cpp(1715).

The assertion fails even with OMP_NUM_THREADS=1. If the second task is removed,
everything runs to completion. If the "omp parallel for" directives are removed
from inside the tasks, once again everything runs fine.
*/

#define N 1024

int main() {
  #pragma omp task
  {
    int i;
    #pragma omp parallel for
    for (i = 0; i < N; i++)
      (void)0;
  }

  #pragma omp task
  {
    int i;
    #pragma omp parallel for
    for (i = 0; i < N; ++i)
      (void)0;
  }

  #pragma omp taskwait

  return 0;
}
