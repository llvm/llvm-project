// RUN: %libomp-compile && env OMP_DISPLAY_AFFINITY=true OMP_PLACES=threads OMP_PROC_BIND=spread,close %libomp-run | %python %S/check.py -c 'CHECK' %s
// REQUIRES: affinity

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "omp_testsuite.h"

int main(int argc, char** argv) {
  omp_set_affinity_format("TESTER: tl:%L at:%a tn:%n nt:%N");
  omp_set_nested(1);
  #pragma omp parallel num_threads(4)
  {
    go_parallel_nthreads(3);
  }
  return get_exit_value();
}

// CHECK: num_threads=4 TESTER: tl:1 at:0 tn:[0-3] nt:4
// CHECK: num_threads=3 TESTER: tl:2 at:[0-3] tn:[0-2] nt:3
// CHECK: num_threads=3 TESTER: tl:2 at:[0-3] tn:[0-2] nt:3
// CHECK: num_threads=3 TESTER: tl:2 at:[0-3] tn:[0-2] nt:3
// CHECK: num_threads=3 TESTER: tl:2 at:[0-3] tn:[0-2] nt:3
