// RUN: %libomp-compile
// RUN: env OMP_PROC_BIND=close OMP_PLACES=threads %libomp-run
// RUN: env OMP_PROC_BIND=close OMP_PLACES=cores %libomp-run
// RUN: env OMP_PROC_BIND=close OMP_PLACES=sockets %libomp-run
// RUN: env KMP_AFFINITY=compact %libomp-run
// RUN: env KMP_AFFINITY=scatter %libomp-run
// REQUIRES: affinity

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

#define DEBUG 0

#if DEBUG
#include <stdarg.h>
#endif

#define BUFFER_SIZE 1024

char buf[BUFFER_SIZE];
#pragma omp threadprivate(buf)

static int debug_printf(const char* format, ...) {
  int retval = 0;
#if DEBUG
  va_list args;
  va_start(args, format);
  retval = vprintf(format, args);
  va_end(args);
#endif
  return retval;
}

static void display_affinity_environment() {
#if DEBUG
  printf("Affinity Environment:\n");
  printf("  OMP_PROC_BIND=%s\n", getenv("OMP_PROC_BIND"));
  printf("  OMP_PLACES=%s\n", getenv("OMP_PLACES"));
  printf("  KMP_AFFINITY=%s\n", getenv("KMP_AFFINITY"));
#endif
}

// Reads in a list of integers into ids array (not going past ids_size)
// e.g., if affinity = "0-4,6,8-10,14,16,17-20,23"
//       then ids = [0,1,2,3,4,6,8,9,10,14,16,17,18,19,20,23]
void list_to_ids(const char* affinity, int* ids, int ids_size) {
  int id, b, e, ids_index;
  char *aff, *begin, *end, *absolute_end;
  aff = strdup(affinity);
  absolute_end = aff + strlen(aff);
  ids_index = 0;
  begin = end = aff;
  while (end < absolute_end) {
    end = begin;
    while (*end != '\0' && *end != ',')
      end++;
    *end = '\0';
    if (strchr(begin, '-') != NULL) {
      // Range
      sscanf(begin, "%d-%d", &b, &e);
    } else {
      // Single Number
      sscanf(begin, "%d", &b);
      e = b;
    }
    for (id = b; id <= e; ++id) {
      ids[ids_index++] = id;
      if (ids_index >= ids_size) {
        free(aff);
        return;
      }
    }
    begin = end + 1;
  }
  free(aff);
}

void check_thread_affinity() {
  int i;
  const char *formats[2] = {"%{thread_affinity}", "%A"};
  for (i = 0; i < sizeof(formats) / sizeof(formats[0]); ++i) {
    omp_set_affinity_format(formats[i]);
    #pragma omp parallel
    {
      int j, k;
      int place = omp_get_place_num();
      int num_procs = omp_get_place_num_procs(place);
      int *ids = (int *)malloc(sizeof(int) * num_procs);
      int *ids2 = (int *)malloc(sizeof(int) * num_procs);
      char buf[256];
      size_t n = omp_capture_affinity(buf, 256, NULL);
      check(n <= 256);
      omp_get_place_proc_ids(place, ids);
      list_to_ids(buf, ids2, num_procs);

      #pragma omp for schedule(static) ordered
      for (k = 0; k < omp_get_num_threads(); ++k) {
        #pragma omp ordered
        {
          debug_printf("Thread %d: captured affinity = %s\n",
                       omp_get_thread_num(), buf);
          for (j = 0; j < num_procs; ++j) {
            debug_printf("Thread %d: ids[%d] = %d ids2[%d] = %d\n",
                         omp_get_thread_num(), j, ids[j], j, ids2[j]);
            check(ids[j] == ids2[j]);
          }
        }
      }

      free(ids);
      free(ids2);
    }
  }
}

int main(int argc, char** argv) {
  omp_set_nested(1);
  display_affinity_environment();
  check_thread_affinity();
  return 0;
}
