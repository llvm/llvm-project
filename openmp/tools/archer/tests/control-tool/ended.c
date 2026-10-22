/*
 * ended.c -- Archer testcase
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
  int var = 0;

#pragma omp parallel
  { omp_control_tool(omp_control_tool_end, 0, NULL); }

#pragma omp parallel num_threads(2) shared(var)
  {
    if (omp_get_thread_num() == 0) {
      var++;
    }

    /* We miss the race due to Archer being paused */
    if (omp_get_thread_num() == 1) {
      var++;
    }
  }

  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK-NOT: ThreadSanitizer: data race
// CHECK-NOT: ThreadSanitizer: reported
// CHECK: DONE
