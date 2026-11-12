// RUN: %libomp-compile
// RUN: env OMP_DISPLAY_AFFINITY=true OMP_AFFINITY_FORMAT='TESTER-ENV: tl:%L tn:%n nt:%N' OMP_NUM_THREADS=8 %libomp-run | %python %S/check.py -c 'CHECK-8' %s

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "omp_testsuite.h"

int main(int argc, char** argv) {
  go_parallel();
  go_parallel();
  return get_exit_value();
}

// CHECK-8: num_threads=8 TESTER-ENV: tl:1 tn:[0-7] nt:8$
