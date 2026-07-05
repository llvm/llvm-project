// RUN: %gdb-compile 2>&1 | tee %t.compile
// RUN: env OMP_SCHEDULE=dynamic,2 %gdb-test -x %S/ompd_icvs.cmd %t 2>&1 | tee
// %t.out | FileCheck %s

#include <omp.h>
#include <stdio.h>
int main(void) {
  omp_set_max_active_levels(3);
  omp_set_dynamic(0);
  omp_set_num_threads(9);
#pragma omp parallel
  {
    omp_set_num_threads(5);
#pragma omp parallel
    {
#pragma omp single
      { printf("Inner: num_thds=%d\n", omp_get_num_threads()); }
    }
#pragma omp barrier
    omp_set_max_active_levels(0);
#pragma omp parallel
    {
#pragma omp single
      { printf("Inner: num_thds=%d\n", omp_get_num_threads()); }
    }
#pragma omp barrier
#pragma omp single
    { printf("Outer: num_thds=%d\n", omp_get_num_threads()); }
  }
  return 0;
}
// CHECK: Loaded OMPD lib successfully!

// CHECK: run-sched-var                   task                       dynamic,2
// CHECK: levels-var                      parallel                   2
// CHECK: active-levels-var               parallel                   2
// CHECK: team-size-var                   parallel                   5

// CHECK: levels-var                      parallel                   2
// CHECK: active-levels-var               parallel                   1
// CHECK: team-size-var                   parallel                   1

// CHECK: levels-var                      parallel                   1
// CHECK: active-levels-var               parallel                   1
// CHECK: team-size-var                   parallel                   9

// CHECK-NOT: Python Exception
// CHECK-NOT: The program is not being run.
// CHECK-NOT: No such file or directory
