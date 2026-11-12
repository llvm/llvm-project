// RUN: %gdb-compile 2>&1 | tee %t.compile
// RUN: env OMP_SCHEDULE=static %gdb-run 2>&1 | tee %t.out | FileCheck %s

#include "../ompt_plugin.h"
#include <omp.h>
#include <stdio.h>
#include <unistd.h>

void bar() {
  int i;
#pragma omp parallel for num_threads(2)
  for (i = 0; i < 10; i++)
    ompd_tool_test(0);
}

void foo() {
  omp_set_max_active_levels(10);
#pragma omp parallel num_threads(2)
  {
    if (omp_get_thread_num() == 0)
      ompd_tool_test(0);
    else
      bar();
  }
}

int main() {
  printf("Process %d started.\n", getpid());
  foo();
  return 0;
}

// CHECK-NOT: OMPT-OMPD mismatch
// CHECK-NOT: Python Exception
// CHECK-NOT: The program is not being run.
