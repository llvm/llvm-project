// RUN: %libomp-compile && env OMP_DISPLAY_AFFINITY=true %libomp-run | %python %S/check.py -c 'CHECK' %s

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char** argv) {
  omp_set_affinity_format("TESTER: tl:%L at:%a tn:%n nt:%N");
  omp_set_nested(1);
  #pragma omp parallel num_threads(1)
  {
    #pragma omp parallel num_threads(2)
    { }
    #pragma omp parallel num_threads(2)
    {
      #pragma omp parallel num_threads(1)
      {
        #pragma omp parallel num_threads(2)
        { }
      }
    }
    #pragma omp parallel num_threads(1)
    { }
  }
  #pragma omp parallel num_threads(2)
  { }
  #pragma omp parallel num_threads(1)
  { }
  return 0;
}

// CHECK: num_threads=1 TESTER: tl:1 at:0 tn:0 nt:1

// CHECK: num_threads=2 TESTER: tl:2 at:[0-9] tn:[01] nt:2

// CHECK: num_threads=1 TESTER: tl:3 at:[0-9] tn:0 nt:1
// CHECK: num_threads=1 TESTER: tl:3 at:[0-9] tn:0 nt:1

// CHECK: num_threads=2 TESTER: tl:4 at:[0-9] tn:[01] nt:2
// CHECK: num_threads=2 TESTER: tl:4 at:[0-9] tn:[01] nt:2

// CHECK: num_threads=1 TESTER: tl:2 at:[0-9] tn:0 nt:1

// CHECK: num_threads=2 TESTER: tl:1 at:[0-9] tn:[01] nt:2

// CHECK: num_threads=1 TESTER: tl:1 at:[0-9] tn:0 nt:1
