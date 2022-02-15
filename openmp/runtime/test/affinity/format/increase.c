// RUN: %libomp-compile && env OMP_DISPLAY_AFFINITY=true %libomp-run | %python %S/check.py -c 'CHECK' %s

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "omp_testsuite.h"

int main(int argc, char** argv) {
  omp_set_affinity_format("TESTER: tl:%L tn:%n nt:%N");
  // should print all for first parallel
  go_parallel_nthreads(4);
  // should print all because of new threads
  go_parallel_nthreads(8);
  // should not print anything here
  go_parallel_nthreads(6);
  // should print all because of new thread
  go_parallel_nthreads(9);
  // should not print anything here
  go_parallel_nthreads(2);

  return get_exit_value();
}

// CHECK: num_threads=4 TESTER: tl:1 tn:[0-3] nt:4
// CHECK: num_threads=8 TESTER: tl:1 tn:[0-7] nt:8
// CHECK: num_threads=6 TESTER: tl:1 tn:[0-5] nt:6
// CHECK: num_threads=9 TESTER: tl:1 tn:[0-8] nt:9
// CHECK: num_threads=2 TESTER: tl:1 tn:[01] nt:2
