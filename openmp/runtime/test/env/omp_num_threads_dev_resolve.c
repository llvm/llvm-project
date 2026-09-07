// RUN: %libomp-compile && \
// RUN:   env OMP_NUM_THREADS_ALL=8 OMP_NUM_THREADS_DEV=64 \
// RUN:   OMP_NUM_THREADS_DEV_0=128 %libomp-run
//
// OpenMP 6.0 non-host precedence: `_DEV_<d>` > `_DEV` > `_ALL` > default.

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern const char *__kmp_resolve_host_env(const char *name);
extern const char *__kmp_resolve_device_env(const char *name, int device_id);

static int check(const char *got, int device_id, const char *expect) {
  if (got == NULL || strcmp(got, expect) != 0) {
    fprintf(stderr, "FAIL: device_id=%d resolved to '%s', expected '%s'\n",
            device_id, got ? got : "(null)", expect);
    return 1;
  }
  return 0;
}

int main(void) {
  int rc = 0;
  (void)omp_get_max_threads();
  rc |= check(__kmp_resolve_host_env("OMP_NUM_THREADS"), -1, "8"); // host: _ALL
  rc |=
      check(__kmp_resolve_device_env("OMP_NUM_THREADS", 0), 0, "128"); // _DEV_0
  rc |= check(__kmp_resolve_device_env("OMP_NUM_THREADS", 1), 1, "64"); // _DEV
  rc |= check(__kmp_resolve_device_env("OMP_NUM_THREADS", 2), 2, "64"); // _DEV
  return rc;
}
