// RUN: %compilexx-run-and-check

#include <stdio.h>
#include <omp.h>

int main(void) {
  int isHost = -1;
  int ParallelLevel1 = -1, ParallelLevel2 = -1;
  int Count = 0;

#pragma omp target parallel for map(tofrom                                     \
                                    : isHost, ParallelLevel1, ParallelLevel2), reduction(+: Count) schedule(static, 1)
  for (int J = 0; J < 10; ++J) {
#pragma omp critical
    {
      isHost = (isHost < 0 || isHost == 0) ? omp_is_initial_device() : isHost;
      ParallelLevel1 = (ParallelLevel1 < 0 || ParallelLevel1 == 1)
                           ? omp_get_level()
                           : ParallelLevel1;
    }
    if (omp_get_thread_num() > 5) {
      int L2;
#pragma omp parallel for schedule(dynamic) lastprivate(L2) reduction(+: Count)
      for (int I = 0; I < 10; ++I) {
        L2 = omp_get_level();
        Count += omp_get_level(); // (10-6)*10*2 = 80
      }
#pragma omp critical
      ParallelLevel2 =
          (ParallelLevel2 < 0 || ParallelLevel2 == 2) ? L2 : ParallelLevel2;
    } else {
      Count += omp_get_level(); // 6 * 1 = 6
    }
  }

  if (isHost < 0) {
    printf("Runtime error, isHost=%d\n", isHost);
  }

  // CHECK: Target region executed on the device
  printf("Target region executed on the %s\n", isHost ? "host" : "device");
  // CHECK: Parallel level in SPMD mode: L1 is 1, L2 is 2
  printf("Parallel level in SPMD mode: L1 is %d, L2 is %d\n", ParallelLevel1,
         ParallelLevel2);
  // Final result of Count is (10-6)(num of loops)*10(num of iterations)*2(par
  // level) + 6(num of iterations) * 1(par level)
  // CHECK: Expected count = 86
  printf("Expected count = %d\n", Count);

  return isHost;
}
