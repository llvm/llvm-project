// RUN: %libomp-compile && env OMP_NUM_THREADS=4 OMP_NUM_THREADS_ALL=8
// %libomp-run
//
// Host OMP_NUM_THREADS beats _ALL; legacy ICV and query API agree.

#include <omp.h>
#include <stdio.h>
#include <string.h>

extern const char *__kmpc_get_resolved_device_env(const char *name,
                                                  int device_id);

int main(void) {
  int max = omp_get_max_threads();
  if (max != 4) {
    fprintf(stderr, "FAIL: omp_get_max_threads()=%d, expected 4\n", max);
    return 1;
  }
  const char *host = __kmpc_get_resolved_device_env("OMP_NUM_THREADS", -1);
  if (host == NULL || strcmp(host, "4") != 0) {
    fprintf(stderr, "FAIL: host query expected '4' got '%s'\n",
            host ? host : "(null)");
    return 1;
  }
  // Sibling `_ALL` must still resolve for non-host devices.
  const char *dev0 = __kmpc_get_resolved_device_env("OMP_NUM_THREADS", 0);
  if (dev0 == NULL || strcmp(dev0, "8") != 0) {
    fprintf(stderr, "FAIL: dev 0 query expected '8' got '%s'\n",
            dev0 ? dev0 : "(null)");
    return 1;
  }
  return 0;
}
