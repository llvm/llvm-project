// RUN: %libomp-compile
// RUN: env KMP_WARNINGS=1 OMP_NUM_THREADS_DEV_abc=10 OMP_NUM_THREADS_ALL=4 \
// RUN:   %libomp-run 2>&1 | FileCheck %s
//
// Malformed `_DEV_<token>` is rejected with a warning, and a sibling
// valid `_ALL` setting is unaffected.

#include <omp.h>
#include <stdio.h>

int main(void) {
  int max = omp_get_max_threads();
  if (max != 4) {
    fprintf(stderr, "FAIL: omp_get_max_threads()=%d, expected 4\n", max);
    return 1;
  }
  printf("DONE\n");
  return 0;
}

// CHECK: {{^OMP: Warning #[0-9]+:.*OMP_NUM_THREADS_DEV_abc.*}}
// CHECK: DONE
