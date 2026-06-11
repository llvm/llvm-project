// RUN: %libomp-compile-and-run
//
// kmp_set_defaults("OMP_NUM_THREADS_ALL=N") must propagate to live threads,
// and an unrelated kmp_set_defaults call must not wipe registry state.

#include <omp.h>
#include <stdio.h>

extern void kmp_set_defaults(const char *);
extern const char *__kmpc_get_resolved_device_env(const char *name,
                                                  int device_id);

int main(void) {
  int rc = 0;
  (void)omp_get_max_threads();

  kmp_set_defaults("OMP_NUM_THREADS_ALL=16");
  if (omp_get_max_threads() != 16) {
    fprintf(stderr, "FAIL: after _ALL=16 expected 16 got %d\n",
            omp_get_max_threads());
    rc = 1;
  }
  const char *q = __kmpc_get_resolved_device_env("OMP_NUM_THREADS", -1);
  if (q == NULL || q[0] != '1' || q[1] != '6' || q[2] != '\0') {
    fprintf(stderr, "FAIL: query host expected '16' got %s\n",
            q ? q : "(null)");
    rc = 1;
  }

  kmp_set_defaults("KMP_BLOCKTIME=200");
  q = __kmpc_get_resolved_device_env("OMP_NUM_THREADS", 0);
  if (q == NULL || q[0] != '1' || q[1] != '6' || q[2] != '\0') {
    fprintf(stderr,
            "FAIL: registry wiped by unrelated kmp_set_defaults; got %s\n",
            q ? q : "(null)");
    rc = 1;
  }
  return rc;
}
