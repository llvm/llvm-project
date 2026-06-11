// RUN: %libomp-compile && env OMP_NUM_THREADS_ALL=8 %libomp-run
//
// OpenMP 6.0: every non-host device falls through to `_ALL` when no
// `_DEV[_d]` is set.

#include <omp.h>
#include <stdio.h>
#include <string.h>

extern const char *__kmpc_get_resolved_device_env(const char *name,
                                                  int device_id);

static int check(int device_id, const char *expect) {
  const char *got =
      __kmpc_get_resolved_device_env("OMP_NUM_THREADS", device_id);
  if (!got || strcmp(got, expect) != 0) {
    fprintf(stderr, "FAIL: device_id=%d got '%s' expected '%s'\n", device_id,
            got ? got : "(null)", expect);
    return 1;
  }
  return 0;
}

int main(void) {
  (void)omp_get_max_threads();
  int rc = 0;
  rc |= check(-1, "8"); // host
  rc |= check(0, "8");
  rc |= check(1, "8");
  rc |= check(2, "8");
  return rc;
}
