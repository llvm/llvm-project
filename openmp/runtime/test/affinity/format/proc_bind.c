// RUN: %libomp-compile && env OMP_DISPLAY_AFFINITY=true OMP_PLACES='{0},{0,1},{0},{0,1},{0},{0,1},{0},{0,1},{0},{0,1},{0}' %libomp-run | %python %S/check.py -c 'CHECK' %s
// REQUIRES: affinity

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "omp_testsuite.h"

int main(int argc, char** argv) {
  omp_set_affinity_format("TESTER: tl:%L tn:%n nt:%N aff:{%A}");
  omp_set_num_threads(8);
  // Initial parallel
  go_parallel_spread();
  go_parallel_spread();
  // Affinity changes here
  go_parallel_close();
  go_parallel_close();
  // Affinity changes here
  go_parallel_master();
  go_parallel_master();
  return get_exit_value();
}

// CHECK: num_threads=8 TESTER: tl:1 tn:[0-7] nt:8 aff:
// CHECK: num_threads=8 TESTER: tl:1 tn:[0-7] nt:8 aff:
// CHECK: num_threads=8 TESTER: tl:1 tn:[0-7] nt:8 aff:
