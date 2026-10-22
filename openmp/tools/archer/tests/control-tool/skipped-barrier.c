/*
 * skipped-barrier.c -- Archer testcase
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
//
// See tools/archer/LICENSE.txt for details.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %libarcher-compile-and-run-race-verbose | FileCheck %s
// REQUIRES: tsan
#include <omp.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  int var = 0;

#pragma omp parallel num_threads(2) shared(var)
  {
    if (omp_get_thread_num() == 0) {
      var++;
    }

    /* TSan detects a race as Archer does not see the barrier */
    omp_control_tool(omp_control_tool_pause, 0, NULL);
#pragma omp barrier
    omp_control_tool(omp_control_tool_start, 0, NULL);

    if (omp_get_thread_num() == 1) {
      var++;
    }
  }

  fprintf(stderr, "DONE\n");
  int error = 1;
  return error;
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK: DONE
// CHECK: ThreadSanitizer: reported {{[1]}} warnings
