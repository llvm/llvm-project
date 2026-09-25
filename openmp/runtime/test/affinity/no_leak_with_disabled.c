// RUN: %libomp-compile -fopenmp
// RUN: %libomp-run
// RUN: env KMP_AFFINITY=disabled %libomp-run

// Check that no memory is leaked with KMP_AFFINITY=disabled.
// The detection is done by ASAN/LSAN.

#include <omp.h>

int main(void) {
#pragma omp parallel
  {
    omp_get_thread_num();
  }
}
