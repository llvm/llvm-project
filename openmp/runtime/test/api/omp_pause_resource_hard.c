// RUN: %libomp-compile-and-run

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#ifdef _WIN32
int setenv(const char *name, const char *value, int overwrite) {
  (void)overwrite; // not used
  errno_t e = _putenv_s(name, value);
  if (e != 0)
    return -1;
  return 0;
}
#endif

int main() {
  int bt;
  char buf[512];
  for (bt = 1; bt <= 50; ++bt) {
    snprintf(buf, sizeof(buf), "%dus", bt);
    setenv("KMP_BLOCKTIME", buf, 1);
    int read_bt = kmp_get_blocktime();
    if (read_bt != bt) {
      fprintf(stderr, "error: kmp_get_blocktime() value (%d) is not %d\n",
              read_bt, bt);
      return EXIT_FAILURE;
    }
    // Call library termination. This should force the library to re-read
    // KMP_BLOCKTIME from the environment during serial initialization
    omp_pause_resource_all(omp_pause_hard);
  }
  printf("Pass\n");
  return EXIT_SUCCESS;
}
