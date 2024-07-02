// RUN: %libomp-compile
// RUN: env KMP_HOT_TEAMS_MODE=0 KMP_HOT_TEAMS_MAX_LEVEL=1 %libomp-run
//
// Force the defaults of:
// KMP_HOT_TEAMS_MODE=0 means free extra threads after parallel
//   involving non-hot team
// KMP_HOT_TEAMS_MAX_LEVEL=1 means only the initial outer team
//   is a hot team.

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
  int a;
  omp_set_max_active_levels(2);
// This nested parallel creates extra threads on the thread pool
#pragma omp parallel num_threads(2)
  {
#pragma omp parallel num_threads(2)
    {
#pragma omp atomic
      a++;
    }
  }

// Causes assert if hidden helper thread tries to allocate from thread pool
// instead of creating new OS threads
#pragma omp parallel num_threads(1)
  {
#pragma omp target nowait
    { a++; }
  }

  return EXIT_SUCCESS;
}
