// RUN: %libomp-compile
// RUN: env KMP_WARNINGS=0 OMP_NUM_THREADS_DEV_abc=10 \
// RUN:   OMP_DEFAULT_DEVICE_ALL=2 OMP_NUM_THREADS_ALL=8 \
// RUN:   %libomp-run 2>&1 | FileCheck %s
//
// KMP_WARNINGS=0 silences device-scope warnings; OMP_NUM_THREADS_ALL=8 applies.

#include <omp.h>
#include <stdio.h>
#include <string.h>

extern const char *__kmpc_get_resolved_device_env(const char *name,
                                                  int device_id);

int main(void) {
  int max = omp_get_max_threads();
  if (max != 8) {
    fprintf(stderr, "FAIL: omp_get_max_threads()=%d, expected 8\n", max);
    return 1;
  }
  const char *host = __kmpc_get_resolved_device_env("OMP_NUM_THREADS", -1);
  if (host == NULL || strcmp(host, "8") != 0) {
    fprintf(stderr, "FAIL: host query expected '8' got '%s'\n",
            host ? host : "(null)");
    return 1;
  }
  printf("DONE\n");
  return 0;
}

// CHECK-NOT: {{^OMP: Warning #[0-9]+:.*OMP_NUM_THREADS_DEV_abc.*}}
// CHECK-NOT: {{^OMP: Warning #[0-9]+:.*OMP_DEFAULT_DEVICE_ALL.*}}
// CHECK: DONE
