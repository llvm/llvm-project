// RUN: %libomptarget-compile-run-and-check-generic

#include <stdio.h>

int omp_get_num_procs(void);

int main() {
  int num_procs;
#pragma omp target map(from : num_procs)
  { num_procs = omp_get_num_procs(); }

  // CHECK: PASS
  if (num_procs > 0)
    printf("PASS\n");
}
