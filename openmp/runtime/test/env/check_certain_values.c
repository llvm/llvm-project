// RUN: %libomp-compile
// RUN: env KMP_FORKJOIN_BARRIER=0,0 %libomp-run
// RUN: env KMP_PLAIN_BARRIER=0,0 %libomp-run
// RUN: env KMP_REDUCTION_BARRIER=0,0 %libomp-run
// RUN: env KMP_ALIGN_ALLOC=7 %libomp-run
// RUN: env KMP_ALIGN_ALLOC=8 %libomp-run
// RUN: env KMP_AFFINITY='explicit,proclist=[0-1222333333333444444]' %libomp-run
// RUN: env KMP_AFFINITY=disabled OMP_DISPLAY_AFFINITY=TRUE %libomp-run
//
// Test that certain environment variable values do not crash the runtime.
#include <omp.h>
#include <stdlib.h>

int a = 0;

int test() {
#pragma omp parallel reduction(+ : a)
  {
    a += omp_get_thread_num();
  }
  if (a == 0) {
    // If the test passes, 'a' should not be zero
    // because we are using reduction on thread numbers.
    return 0;
  }
  return 1;
}

int main(int argc, char **argv) {
  int status = EXIT_SUCCESS;
  if (!test()) {
    status = EXIT_FAILURE;
  }
  return status;
}
