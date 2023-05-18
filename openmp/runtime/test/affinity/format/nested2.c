// RUN: %libomp-compile && env OMP_DISPLAY_AFFINITY=true OMP_PLACES=threads OMP_PROC_BIND=spread,close KMP_HOT_TEAMS_MAX_LEVEL=2 %libomp-run | %python %S/check.py -c 'CHECK' %s

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "omp_testsuite.h"

// Currently, KMP_HOT_TEAMS_MAX_LEVEL has to be equal to the
// nest depth for intuitive behavior
int main(int argc, char** argv) {
  omp_set_affinity_format("TESTER: tl:%L tn:%n nt:%N");
  omp_set_nested(1);
  #pragma omp parallel num_threads(4)
  {
    go_parallel_nthreads(3);
    go_parallel_nthreads(3);
  }
  go_parallel_nthreads(4);
  return get_exit_value();
}

// CHECK: num_threads=4 TESTER: tl:1 tn:[0-3] nt:4
// CHECK: num_threads=3 TESTER: tl:2 tn:[0-2] nt:3
// CHECK: num_threads=3 TESTER: tl:2 tn:[0-2] nt:3
// CHECK: num_threads=3 TESTER: tl:2 tn:[0-2] nt:3
// CHECK: num_threads=3 TESTER: tl:2 tn:[0-2] nt:3
// CHECK: num_threads=4 TESTER: tl:1 tn:[0-3] nt:4
