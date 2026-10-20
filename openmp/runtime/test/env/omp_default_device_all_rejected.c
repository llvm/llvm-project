// RUN: %libomp-compile
// RUN: env KMP_WARNINGS=1 OMP_DEFAULT_DEVICE_ALL=2 %libomp-run 2>&1 \
// RUN:   | FileCheck %s
//
// Suffix on global-scope base must be rejected with a warning AND not
// reach the ICV.

#include <omp.h>
#include <stdio.h>

int main(void) {
  (void)omp_get_max_threads();
  int dd = omp_get_default_device();
  if (dd == 2) {
    fprintf(stderr, "FAIL: OMP_DEFAULT_DEVICE_ALL=2 was applied; got %d\n", dd);
    return 1;
  }
  printf("DONE\n");
  return 0;
}

// CHECK: {{^OMP: Warning #[0-9]+:.*OMP_DEFAULT_DEVICE_ALL.*}}
// CHECK: DONE
