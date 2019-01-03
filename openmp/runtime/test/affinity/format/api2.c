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

#if defined(_WIN32)
#define snprintf _snprintf
#endif

#define BUFFER_SIZE 1024

int main(int argc, char** argv) {
  char buf[BUFFER_SIZE];
  size_t needed, length;
  const char* format = "tl:%L tn:%n nt:%N an:%a";
  const char* second_format = "nesting_level:%{nesting_level} thread_num:%{thread_num} num_threads:%{num_threads} ancestor_tnum:%{ancestor_tnum}";

  length = strlen(format);
  omp_set_affinity_format(format);

  needed = omp_get_affinity_format(buf, BUFFER_SIZE);
  check(streqls(buf, format));
  check(needed == length)

  // Check that it is truncated properly
  omp_get_affinity_format(buf, 5);
  check(streqls(buf, "tl:%"));

  #pragma omp parallel
  {
    char my_buf[512];
    char supposed[512];
    int tl, tn, nt, an;
    size_t needed, needed2;
    tl = omp_get_level();
    tn = omp_get_thread_num();
    nt = omp_get_num_threads();
    an = omp_get_ancestor_thread_num(omp_get_level()-1);
    needed = omp_capture_affinity(my_buf, 512, NULL);
    needed2 = (size_t)snprintf(supposed, 512, "tl:%d tn:%d nt:%d an:%d", tl, tn, nt, an);
    check(streqls(my_buf, supposed));
    check(needed == needed2);
    // Check that it is truncated properly
    supposed[4] = '\0';
    omp_capture_affinity(my_buf, 5, NULL);
    check(streqls(my_buf, supposed));

    needed = omp_capture_affinity(my_buf, 512, second_format);
    needed2 = (size_t)snprintf(supposed, 512, "nesting_level:%d thread_num:%d num_threads:%d ancestor_tnum:%d", tl, tn, nt, an);
    check(streqls(my_buf, supposed));
    check(needed == needed2);

    // Check that it is truncated properly
    supposed[25] = '\0';
    omp_capture_affinity(my_buf, 26, second_format);
    check(streqls(my_buf, supposed));
  }

  #pragma omp parallel num_threads(4)
  {
    omp_display_affinity(NULL);
    omp_display_affinity(second_format);
  }

  return 0;
}

// CHECK: num_threads=4 tl:[0-9]+ tn:[0-9]+ nt:[0-9]+ an:[0-9]+
// CHECK: num_threads=4 nesting_level:[0-9]+ thread_num:[0-9]+ num_threads:[0-9]+ ancestor_tnum:[0-9]+
