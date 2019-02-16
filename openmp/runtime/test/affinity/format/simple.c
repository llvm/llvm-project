// RUN: %libomp-compile
// RUN: env OMP_DISPLAY_AFFINITY=false %libomp-run | %python %S/check.py -c 'NOTHING' %s
// RUN: env OMP_DISPLAY_AFFINITY=true OMP_NUM_THREADS=1 %libomp-run | %python %S/check.py -c 'CHECK' %s
// RUN: env OMP_DISPLAY_AFFINITY=true OMP_NUM_THREADS=2 %libomp-run | %python %S/check.py -c 'CHECK-2' %s
// RUN: env OMP_DISPLAY_AFFINITY=true OMP_NUM_THREADS=3 %libomp-run | %python %S/check.py -c 'CHECK-3' %s
// RUN: env OMP_DISPLAY_AFFINITY=true OMP_NUM_THREADS=4 %libomp-run | %python %S/check.py -c 'CHECK-4' %s
// RUN: env OMP_DISPLAY_AFFINITY=true OMP_NUM_THREADS=8 %libomp-run | %python %S/check.py -c 'CHECK-8' %s

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char** argv) {
  omp_set_affinity_format("TESTER: tl:%L tn:%n nt:%N");
  #pragma omp parallel
  { }
  #pragma omp parallel
  { }
  return 0;
}

// NOTHING: NO_OUTPUT
// CHECK: num_threads=1 TESTER: tl:1 tn:0 nt:1
// CHECK-2: num_threads=2 TESTER: tl:1 tn:[01] nt:2
// CHECK-3: num_threads=3 TESTER: tl:1 tn:[0-2] nt:3
// CHECK-4: num_threads=4 TESTER: tl:1 tn:[0-3] nt:4
// CHECK-8: num_threads=8 TESTER: tl:1 tn:[0-7] nt:8
