// RUN: %compile-run-and-check

#include <omp.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  int MaxThreadsL1 = -1, MaxThreadsL2 = -1;

#pragma omp declare reduction(unique:int                                       \
                              : omp_out = (omp_in == 1 ? omp_in : omp_out))    \
    initializer(omp_priv = -1)

  // Non-SPMD mode.
#pragma omp target teams map(MaxThreadsL1, MaxThreadsL2) thread_limit(32)      \
    num_teams(1)
  {
    MaxThreadsL1 = omp_get_max_threads();
#pragma omp parallel reduction(unique : MaxThreadsL2)
    { MaxThreadsL2 = omp_get_max_threads(); }
  }

  //FIXME: This Non-SPMD kernel will have 32 active threads due to
  //       thread_limit. However, Non-SPMD MaxThreadsL1 is the total number of
  //       threads in block (64 in this case), which translates to worker
  //       threads + WARP_SIZE for Non-SPMD kernels and worker threads for SPMD
  //       kernels. According to the spec, omp_get_max_threads must return the
  //       max active threads possible between the two kernel types.

  // CHECK: Non-SPMD MaxThreadsL1 = 64
  printf("Non-SPMD MaxThreadsL1 = %d\n", MaxThreadsL1);
  // CHECK: Non-SPMD MaxThreadsL2 = 1
  printf("Non-SPMD MaxThreadsL2 = %d\n", MaxThreadsL2);

  // SPMD mode with full runtime
  MaxThreadsL2 = -1;
#pragma omp target parallel reduction(unique : MaxThreadsL2)
  { MaxThreadsL2 = omp_get_max_threads(); }

  // CHECK: SPMD with full runtime MaxThreadsL2 = 1
  printf("SPMD with full runtime MaxThreadsL2 = %d\n", MaxThreadsL2);

  // SPMD mode without runtime
  MaxThreadsL2 = -1;
#pragma omp target parallel for reduction(unique : MaxThreadsL2)
  for (int I = 0; I < 2; ++I) {
    MaxThreadsL2 = omp_get_max_threads();
  }

  // CHECK: SPMD without runtime MaxThreadsL2 = 1
  printf("SPMD without runtime MaxThreadsL2 = %d\n", MaxThreadsL2);

  return 0;
}
