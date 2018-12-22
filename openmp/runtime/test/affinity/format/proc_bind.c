// RUN: %libomp-compile && env OMP_DISPLAY_AFFINITY=true OMP_PLACES='{0},{0,1},{0},{0,1},{0},{0,1},{0},{0,1},{0},{0,1},{0}' %libomp-run | %python %S/check.py -c 'CHECK' %s
// REQUIRES: affinity

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char** argv) {
  omp_set_affinity_format("TESTER: tl:%L tn:%n nt:%N aff:{%A}");
  omp_set_num_threads(8);
  // Initial parallel
  #pragma omp parallel proc_bind(spread)
  { }
  #pragma omp parallel proc_bind(spread)
  { }
  // Affinity changes here
  #pragma omp parallel proc_bind(close)
  { }
  #pragma omp parallel proc_bind(close)
  { }
  // Affinity changes here
  #pragma omp parallel proc_bind(master)
  { }
  #pragma omp parallel proc_bind(master)
  { }
  return 0;
}

// CHECK: num_threads=8 TESTER: tl:1 tn:[0-7] nt:8 aff:
// CHECK: num_threads=8 TESTER: tl:1 tn:[0-7] nt:8 aff:
// CHECK: num_threads=8 TESTER: tl:1 tn:[0-7] nt:8 aff:
