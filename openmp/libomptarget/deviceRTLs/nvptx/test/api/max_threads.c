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

  // CHECK: Non-SPMD MaxThreadsL1 = 32
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
