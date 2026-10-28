// RUN: %libomp-compile
// RUN: env KMP_WARNINGS=1 OMP_TARGET_OFFLOAD_DEV_0=mandatory \
// RUN:   OMP_TOOL_LIBRARIES_ALL=libfoo.so \
// RUN:   OMP_CANCELLATION_DEV=true \
// RUN:   OMP_NUM_THREADS_ALL=8 \
// RUN:   %libomp-run 2>&1 | FileCheck %s
//
// Denylist: suffixed global OMP_* warns; OMP_NUM_THREADS_ALL still applies.

#include <omp.h>
#include <stdio.h>

int main(void) {
  int max = omp_get_max_threads();
  if (max != 8) {
    fprintf(stderr, "FAIL: omp_get_max_threads()=%d, expected 8\n", max);
    return 1;
  }
  printf("DONE\n");
  return 0;
}

// CHECK-DAG: {{^OMP: Warning #[0-9]+:.*OMP_TARGET_OFFLOAD_DEV_0.*}}
// CHECK-DAG: {{^OMP: Warning #[0-9]+:.*OMP_TOOL_LIBRARIES_ALL.*}}
// CHECK-DAG: {{^OMP: Warning #[0-9]+:.*OMP_CANCELLATION_DEV.*}}
// CHECK: DONE
