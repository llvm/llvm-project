// RUN: %libomp-compile
// RUN: env KMP_WARNINGS=1 OMP_NUM_THREADS_DEV_2147483647=10 \
// RUN:   OMP_NUM_THREADS_DEV_99999999999=20 \
// RUN:   OMP_NUM_THREADS_DEV0=30 OMP_NUM_THREADS_ALLEY=40 \
// RUN:   OMP_NUM_THREADS_ALL=4 \
// RUN:   %libomp-run 2>&1 | FileCheck %s
//
// (a) `_DEV_<huge>` rejected with a warning;
// (b) similar-looking names (`_DEV0`, `_ALLEY`) silently ignored, no warning;
// (c) sibling valid `_ALL=4` still applies.

#include <omp.h>
#include <stdio.h>
#include <string.h>

extern const char *__kmpc_get_resolved_device_env(const char *name,
                                                  int device_id);

int main(void) {
  if (omp_get_max_threads() != 4) {
    fprintf(stderr, "FAIL: host got %d, expected 4\n", omp_get_max_threads());
    return 1;
  }
  const char *q = __kmpc_get_resolved_device_env("OMP_NUM_THREADS", 2147483646);
  if (q == NULL || strcmp(q, "4") != 0) {
    fprintf(stderr, "FAIL: large valid id expected '4', got %s\n",
            q ? q : "(null)");
    return 1;
  }
  printf("DONE\n");
  return 0;
}

// CHECK:     {{^OMP: Warning #[0-9]+:.*OMP_NUM_THREADS_DEV_2147483647.*}}
// CHECK:     {{^OMP: Warning #[0-9]+:.*OMP_NUM_THREADS_DEV_99999999999.*}}
// CHECK-NOT: {{^OMP: Warning #[0-9]+:.*OMP_NUM_THREADS_DEV0.*}}
// CHECK-NOT: {{^OMP: Warning #[0-9]+:.*OMP_NUM_THREADS_ALLEY.*}}
// CHECK:     DONE
