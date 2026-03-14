// RUN: %gdb-compile 2>&1 | tee %t.compile
// RUN: %gdb-test -x %s.cmd %t 2>&1 | tee %t.out | FileCheck %s
// RUN: %gdb-test -x %s.cmd2 %t 2>&1 | tee %t.out2 \
// RUN:                              | FileCheck --check-prefix CMD2 %s
#include <omp.h>
#include <stdio.h>

int main() {
  omp_set_num_threads(4);
#pragma omp parallel
  { printf("Parallel level 1, thread num = %d.\n", omp_get_thread_num()); }
  return 0;
}
// CHECK-NOT: Failed
// CHECK-NOT: Skip

// CMD2: Run 'ompd init' before running any of the ompd commands
// CMD2: Error in Initialization
