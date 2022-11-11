// RUN: %gdb-compile 2>&1 | tee %t.compile
// RUN: %gdb-test -x %S/ompd_parallel.cmd %t 2>&1 | tee %t.out | FileCheck %s

#include <omp.h>
#include <stdio.h>

int main() {
  omp_set_max_active_levels(3);
  omp_set_num_threads(7);
#pragma omp parallel
  {
    omp_set_num_threads(5);
#pragma omp parallel
    {
      omp_set_num_threads(3);
#pragma omp parallel
      { printf("In nested level:3, team size = %d.\n", omp_get_num_threads()); }

      printf("In nested level:2, team size = %d.\n", omp_get_num_threads());
    }
    printf("In nested level:1, team size = %d.\n", omp_get_num_threads());
  }

  return 0;
}

// CHECK: Loaded OMPD lib successfully!
// CHECK: Nesting Level 3: Team Size: 3
// CHECK: ompd_parallel.c{{[ ]*}}:16
// CHECK: Nesting Level 2: Team Size: 5
// CHECK: ompd_parallel.c{{[ ]*}}:13
// CHECK: Nesting Level 1: Team Size: 7
// CHECK: ompd_parallel.c{{[ ]*}}:10

// CHECK: Nesting Level 2: Team Size: 5
// CHECK: ompd_parallel.c{{[ ]*}}:13
// CHECK: Nesting Level 1: Team Size: 7
// CHECK: ompd_parallel.c{{[ ]*}}:10

// CHECK: Nesting Level 1: Team Size: 7
// CHECK: ompd_parallel.c{{[ ]*}}:10

// CHECK-NOT: Python Exception
// CHECK-NOT: The program is not being run.
// CHECK-NOT: No such file or directory
