// RUN: %compilexx-run-and-check

#include <stdio.h>
#include <omp.h>

int main(void) {
  int isHost = -1;
  int ParallelLevel1, ParallelLevel2 = -1;

#pragma omp target parallel map(from: isHost, ParallelLevel1, ParallelLevel2)
  {
    isHost = omp_is_initial_device();
    ParallelLevel1 = omp_get_level();
#pragma omp parallel for schedule(dynamic) lastprivate(ParallelLevel2)
    for (int I = 0; I < 10; ++I)
      ParallelLevel2 = omp_get_level();
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
