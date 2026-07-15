// RUN: %libomp-compile
// RUN: env OMP_DISPLAY_AFFINITY=TRUE OMP_NUM_THREADS=4 OMP_PLACES='{0,1},{2,3},{4,5},{6,7}' %libomp-run | %python %S/check.py -c 'CHECK' %s

// Affinity Display examples
#include <stdio.h>
#include <stdlib.h> // also null is in <stddef.h>
#include <stddef.h>
#include <omp.h>
#include <string.h>

// ENVIRONMENT
// OMP_DISPLAY_AFFINITY=TRUE
// OMP_NUM_THREADS=4
// OMP_PLACES='{0,1},{2,3},{4,5},{6,7}'

// CHECK: num_threads=1 OMP: pid [0-9]+ tid [0-9]+ thread [0-4] bound to OS proc set \{([0-7])|(0,1)|(undefined)\}
// CHECK: num_threads=4 Thread id [0-3] reporting in
// CHECK: num_threads=4 OMP: pid [0-9]+ tid [0-9]+ thread [0-4] bound to OS proc set \{([0-7])|([0246],[1357])|(undefined)\}
// CHECK: num_threads=1 Default Affinity Format is:
// CHECK: num_threads=1 Affinity Format set to: host=%20H tid=%0.4n binds_to=%A
// CHECK: num_threads=4 tid=[0-3] affinity:host=[a-zA-Z0-9_.-]+[ ]+tid=000[0-4][ ]+binds_to=(([0-7])|([0246],[1357])|(undefined))

#define FORMAT_STORE 80
#define BUFFER_STORE 80

int main(int argc, char** argv) {
  int i, n, tid, max_req_store = 0;
  size_t nchars;
  char default_format[FORMAT_STORE];
  char my_format[] = "host=%20H tid=%0.4n binds_to=%A";
  char **buffer;

  // CODE SEGMENT 1 AFFINITY DISPLAY
  omp_display_affinity(NULL);

  // OMP_DISPLAY_AFFINITY=TRUE,
  // Affinity reported for 1 parallel region
  #pragma omp parallel
  {
    printf("Thread id %d reporting in.\n", omp_get_thread_num());
  }

  // Get and Display Default Affinity Format
  nchars = omp_get_affinity_format(default_format, (size_t)FORMAT_STORE);
  printf("Default Affinity Format is: %s\n", default_format);

  if (nchars > FORMAT_STORE) {
    printf("Caution: Reported Format is truncated. Increase\n");
    printf(" FORMAT_STORE by %d.\n", (int)nchars - FORMAT_STORE);
  }

  // Set Affinity Format
  omp_set_affinity_format(my_format);
  printf("Affinity Format set to: %s\n", my_format);

  // CODE SEGMENT 3 CAPTURE AFFINITY
  // Set up buffer for affinity of n threads
  n = omp_get_max_threads();
  buffer = (char **)malloc(sizeof(char *) * n);
  for (i = 0; i < n; i++) {
    buffer[i] = (char *)malloc(sizeof(char) * BUFFER_STORE);
  }

  // Capture Affinity using Affinity Format set above.
  // Use critical reduction to check size of buffer areas
  #pragma omp parallel private(tid, nchars)
  {
    tid = omp_get_thread_num();
    nchars = omp_capture_affinity(buffer[tid], (size_t)BUFFER_STORE, NULL);
    #pragma omp critical
    {
      if (nchars > max_req_store)
        max_req_store = nchars;
    }
  }

  for (i = 0; i < n; i++) {
    printf("tid=%d affinity:%s:\n", i, buffer[i]);
  }
  // for 4 threads with OMP_PLACES='{0,1},{2,3},{4,5},{6,7}'
  // host=%20H tid=%0.4n binds_to=%A
  // host=<hostname> tid=0000 binds_to=0,1
  // host=<hostname> tid=0001 binds_to=2,3
  // host=<hostname> tid=0002 binds_to=4,5
  // host=<hostname> tid=0003 binds_to=6,7

  if (max_req_store > BUFFER_STORE) {
    printf("Caution: Affinity string truncated. Increase\n");
    printf(" BUFFER_STORE by %d\n", max_req_store - BUFFER_STORE);
  }
  return 0;
}
