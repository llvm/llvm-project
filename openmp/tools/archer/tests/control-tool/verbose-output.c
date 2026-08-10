/*
 * verbose-output.c -- Archer testcase
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
//
// See tools/archer/LICENSE.txt for details.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %libarcher-compile-and-run-verbose | FileCheck %s
// REQUIRES: tsan
#include "ompt/ompt-signal.h"
#include <omp.h>
#include <stdio.h>

void foo() {

  int x = 0, y = 2, sem = 0;

#pragma omp task depend(inout : x) shared(x, sem)
  {
    omp_control_tool(omp_control_tool_pause, 0, NULL);
    OMPT_SIGNAL(sem);
    x++; // 1st Child Task
  }

#pragma omp task shared(y, sem)
  {
    omp_control_tool(omp_control_tool_pause, 0, NULL);
    OMPT_SIGNAL(sem);
    y--; // 2nd child task
  }

  OMPT_WAIT(sem, 2);
#pragma omp taskwait depend(in : x) // 1st taskwait

  printf("x=%d\n", x);

#pragma omp taskwait // 2nd taskwait

  printf("y=%d\n", y);
}

int main(int argc, char *argv[]) {
  /* Try out different OpenMP constructs to check whether Tsan IgnoreBegin/Ends
   * are matched correctly */
#pragma omp parallel num_threads(2)
  {
    omp_control_tool(omp_control_tool_start, 0, NULL);
    omp_control_tool(omp_control_tool_pause, 0, NULL);
    omp_control_tool(omp_control_tool_start, 0, NULL);
  }

#pragma omp parallel num_threads(2)
  { omp_control_tool(omp_control_tool_pause, 0, NULL); }

#pragma omp parallel num_threads(1)
  { omp_control_tool(omp_control_tool_pause, 0, NULL); }

#pragma omp parallel num_threads(2)
  {
    omp_control_tool(omp_control_tool_pause, 0, NULL);
    omp_control_tool(omp_control_tool_start, 0, NULL);
    omp_control_tool(omp_control_tool_end, 0, NULL);
  }

  foo();

  fprintf(stderr, "DONE\n");
  return 0;
}
// CHECK-NOT: One of the following ignores was not ended
// CHECK-NOT: finished with ignores enabled
// CHECK-NOT: ThreadSanitizer: data race
// CHECK-NOT: ThreadSanitizer: reported
// CHECK-NOT: Warning: please export TSAN_OPTIONS
// CHECK: DONE
// CHECK: [Archer] Paused operation
// CHECK: [Archer] Started operation
// CHECK: [Archer] Ended operation
