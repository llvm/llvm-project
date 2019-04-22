// RUN: %compilexx-run-and-check

#include <stdio.h>
#include <omp.h>

int main(void) {
  int isHost = -1;
  int ParallelLevel1 = -1, ParallelLevel2 = -1;

#pragma omp target parallel for map(tofrom                                     \
                                    : isHost, ParallelLevel1, ParallelLevel2)
  for (int J = 0; J < 10; ++J) {
#pragma omp critical
    {
      isHost = (isHost < 0 || isHost == omp_is_initial_device())
                   ? omp_is_initial_device()
                   : 1;
      ParallelLevel1 =
          (ParallelLevel1 < 0 || ParallelLevel1 == 1) ? omp_get_level() : 2;
    }
    int L2;
#pragma omp parallel for schedule(dynamic) lastprivate(L2)
    for (int I = 0; I < 10; ++I)
      L2 = omp_get_level();
#pragma omp critical
    ParallelLevel2 = (ParallelLevel2 < 0 || ParallelLevel2 == 2) ? L2 : 1;
  }

  if (isHost < 0) {
    printf("Runtime error, isHost=%d\n", isHost);
  }

  // CHECK: Target region executed on the device
  printf("Target region executed on the %s\n", isHost ? "host" : "device");
  // CHECK: Parallel level in SPMD mode: L1 is 1, L2 is 2
  printf("Parallel level in SPMD mode: L1 is %d, L2 is %d\n", ParallelLevel1,
         ParallelLevel2);

  return isHost;
}
