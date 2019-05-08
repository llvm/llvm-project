// RUN: %compile-run-and-check

#include <omp.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  int ThreadLimitL0 = -1, ThreadLimitL1 = -1, ThreadLimitL2 = -1;

#pragma omp declare reduction(unique64:int                                     \
                              : omp_out = (omp_in == 64 ? omp_in : omp_out))   \
    initializer(omp_priv = -1)
#pragma omp declare reduction(unique32:int                                     \
                              : omp_out = (omp_in == 32 ? omp_in : omp_out))   \
    initializer(omp_priv = -1)

  // Non-SPMD mode.
#pragma omp target teams map(ThreadLimitL0, ThreadLimitL1, ThreadLimitL2)      \
    thread_limit(64) num_teams(1)
  {
    ThreadLimitL0 = omp_get_thread_limit();
#pragma omp parallel reduction(unique64                                        \
                               : ThreadLimitL1, ThreadLimitL2) num_threads(32)
    {
      ThreadLimitL1 = omp_get_thread_limit();
#pragma omp parallel reduction(unique64 : ThreadLimitL2)
      { ThreadLimitL2 = omp_get_thread_limit(); }
    }
  }

  // CHECK: Non-SPMD ThreadLimitL0 = 64
  printf("Non-SPMD ThreadLimitL0 = %d\n", ThreadLimitL0);
  // CHECK: Non-SPMD ThreadLimitL1 = 64
  printf("Non-SPMD ThreadLimitL1 = %d\n", ThreadLimitL1);
  // CHECK: Non-SPMD ThreadLimitL2 = 64
  printf("Non-SPMD ThreadLimitL2 = %d\n", ThreadLimitL2);

  // SPMD mode with full runtime
  ThreadLimitL1 = -1;
  ThreadLimitL2 = -1;
#pragma omp target parallel reduction(unique32                                 \
                                      : ThreadLimitL1, ThreadLimitL2)          \
    num_threads(32)
  {
    ThreadLimitL1 = omp_get_thread_limit();
#pragma omp parallel reduction(unique32 : ThreadLimitL2)
    { ThreadLimitL2 = omp_get_thread_limit(); }
  }

  // CHECK: SPMD with full runtime ThreadLimitL1 = 32
  printf("SPMD with full runtime ThreadLimitL1 = %d\n", ThreadLimitL1);
  // CHECK: SPMD with full runtime ThreadLimitL2 = 32
  printf("SPMD with full runtime ThreadLimitL2 = %d\n", ThreadLimitL2);

  // SPMD mode without runtime
  ThreadLimitL1 = -1;
  ThreadLimitL2 = -1;
#pragma omp target parallel for reduction(unique32                             \
                                          : ThreadLimitL1, ThreadLimitL2)      \
    num_threads(32)
  for (int I = 0; I < 2; ++I) {
    ThreadLimitL1 = omp_get_thread_limit();
#pragma omp parallel reduction(unique32 : ThreadLimitL2)
    { ThreadLimitL2 = omp_get_thread_limit(); }
  }

  // CHECK: SPMD without runtime ThreadLimitL1 = 32
  printf("SPMD without runtime ThreadLimitL1 = %d\n", ThreadLimitL1);
  // CHECK: SPMD without runtime ThreadLimitL2 = 32
  printf("SPMD without runtime ThreadLimitL2 = %d\n", ThreadLimitL2);

  return 0;
}
