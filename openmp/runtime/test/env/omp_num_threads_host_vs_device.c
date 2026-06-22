// RUN: %libomp-compile && \
// RUN:   env OMP_NUM_THREADS=4 OMP_NUM_THREADS_ALL=8 \
// RUN:   OMP_NUM_THREADS_DEV_0=128 %libomp-run
//
// Host and non-host devices resolve through separate functions.

#include <omp.h>
#include <stdio.h>
#include <string.h>

extern const char *__kmp_resolve_host_env(const char *name);
extern const char *__kmp_resolve_device_env(const char *name, int device_id);

int main(void) {
  (void)omp_get_max_threads();

  // Host uses the unsuffixed value, not _ALL.
  const char *host = __kmp_resolve_host_env("OMP_NUM_THREADS");
  if (host == NULL || strcmp(host, "4") != 0)
    return 1;

  // Device 0 uses _DEV_0; device 1 falls through to _ALL.
  const char *dev0 = __kmp_resolve_device_env("OMP_NUM_THREADS", 0);
  if (dev0 == NULL || strcmp(dev0, "128") != 0)
    return 2;
  const char *dev1 = __kmp_resolve_device_env("OMP_NUM_THREADS", 1);
  if (dev1 == NULL || strcmp(dev1, "8") != 0)
    return 3;

  return 0;
}
