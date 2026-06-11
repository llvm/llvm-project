// RUN: %libomp-compile && env OMP_NUM_THREADS_ALL=8 OMP_NUM_THREADS_DEV=64
// OMP_NUM_THREADS_DEV_0=128 %libomp-run
//
// OpenMP 6.0 non-host precedence: `_DEV_<d>` > `_DEV` > `_ALL` > default.

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern const char *__kmpc_get_resolved_device_env(const char *name,
                                                  int device_id);

static int check(const char *name, int device_id, const char *expect) {
  const char *got = __kmpc_get_resolved_device_env(name, device_id);
  if (got == NULL) {
    fprintf(stderr, "FAIL: %s device_id=%d resolved to NULL, expected %s\n",
            name, device_id, expect);
    return 1;
  }
  if (strcmp(got, expect) != 0) {
    fprintf(stderr, "FAIL: %s device_id=%d resolved to '%s', expected '%s'\n",
            name, device_id, got, expect);
    return 1;
  }
  return 0;
}

int main(void) {
  int rc = 0;
  (void)omp_get_max_threads();
  rc |= check("OMP_NUM_THREADS", -1, "8"); // host: _ALL
  rc |= check("OMP_NUM_THREADS", 0, "128"); // _DEV_0 wins
  rc |= check("OMP_NUM_THREADS", 1, "64"); // _DEV
  rc |= check("OMP_NUM_THREADS", 2, "64"); // _DEV (no _DEV_2)

  // Strict host-sentinel contract: only -1 is host.
  if (__kmpc_get_resolved_device_env("OMP_NUM_THREADS", -2) != NULL) {
    fprintf(stderr, "FAIL: device_id=-2 must return NULL\n");
    rc = 1;
  }
  return rc;
}
