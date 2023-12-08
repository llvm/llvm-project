// RUN: %libomp-compile-and-run
// RUN: %libomp-run | %python %S/check.py -c 'CHECK' %s

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define XSTR(x) #x
#define STR(x) XSTR(x)

#define streqls(s1, s2) (!strcmp(s1, s2))

#define check(condition)                                                       \
  if (!(condition)) {                                                          \
    fprintf(stderr, "error: %s: %d: " STR(condition) "\n", __FILE__,           \
            __LINE__);                                                         \
    exit(1);                                                                   \
  }

#define BUFFER_SIZE 1024

int main(int argc, char** argv) {
  char buf[BUFFER_SIZE];
  size_t needed;

  omp_set_affinity_format("0123456789");

  needed = omp_get_affinity_format(buf, BUFFER_SIZE);
  check(streqls(buf, "0123456789"));
  check(needed == 10)

  // Check that it is truncated properly
  omp_get_affinity_format(buf, 5);
  check(streqls(buf, "0123"));

  #pragma omp parallel
  {
    char my_buf[512];
    size_t needed = omp_capture_affinity(my_buf, 512, NULL);
    check(streqls(my_buf, "0123456789"));
    check(needed == 10);
    // Check that it is truncated properly
    omp_capture_affinity(my_buf, 5, NULL);
    check(streqls(my_buf, "0123"));
  }

  #pragma omp parallel num_threads(4)
  {
    omp_display_affinity(NULL);
  }

  return 0;
}

// CHECK: num_threads=4 0123456789
