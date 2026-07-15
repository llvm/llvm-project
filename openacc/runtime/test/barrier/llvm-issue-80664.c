// RUN: %libomp-compile
// RUN: env OMP_WAIT_POLICY=passive \
// RUN:     KMP_FORKJOIN_BARRIER_PATTERN='linear,linear' %libomp-run
// RUN: env OMP_WAIT_POLICY=passive \
// RUN:     KMP_FORKJOIN_BARRIER_PATTERN='tree,tree' %libomp-run
// RUN: env OMP_WAIT_POLICY=passive \
// RUN:     KMP_FORKJOIN_BARRIER_PATTERN='hyper,hyper' %libomp-run
// RUN: env OMP_WAIT_POLICY=passive \
// RUN:     KMP_FORKJOIN_BARRIER_PATTERN='dist,dist' %libomp-run
//
// LLVM ISSUE 80664: https://github.com/llvm/llvm-project/issues/80664
//
// Distributed barrier + OMP_WAIT_POLICY=passive hangs in library termination
// Reason: the resume logic in __kmp_free_team() was faulty and, when checking
// for sleep status, didn't look at correct location for distributed barrier.

#include <stdio.h>
#include <stdlib.h>

int a = 0;

void test_omp_barrier() {
#pragma omp parallel
  {
#pragma omp task
    {
#pragma omp atomic
      a++;
    }
  }
}

int main() {
  test_omp_barrier();
  printf("a = %d\n", a);
  return EXIT_SUCCESS;
}
