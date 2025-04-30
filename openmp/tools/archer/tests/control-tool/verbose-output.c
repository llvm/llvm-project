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
#include <omp.h>
#include <stdio.h>

int main(int argc, char *argv[]) {

#pragma omp parallel num_threads(2)
{
    omp_control_tool(omp_control_tool_pause, 0, NULL);
    omp_control_tool(omp_control_tool_start, 0, NULL);
}

#pragma omp parallel num_threads(2)
{
    omp_control_tool(omp_control_tool_pause, 0, NULL);
}

#pragma omp parallel num_threads(1)
{
    omp_control_tool(omp_control_tool_pause, 0, NULL);
}

#pragma omp parallel num_threads(2)
{
    omp_control_tool(omp_control_tool_pause, 0, NULL);
    omp_control_tool(omp_control_tool_start, 0, NULL);
    omp_control_tool(omp_control_tool_end, 0, NULL);
}

  fprintf(stderr, "DONE\n");
  return 0;
}
// CHECK-NOT: One of the following ignores was not ended
// CHECK-NOT: ThreadSanitizer: data race
// CHECK-NOT: ThreadSanitizer: reported
// CHECK-NOT: Warning: please export TSAN_OPTIONS
// CHECK: DONE
// CHECK: [Archer] Paused operation
// CHECK: [Archer] Started operation
// CHECK: [Archer] Ended operation

