/*
 * parallel-reduction.c -- Archer testcase
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
//
// See tools/archer/LICENSE.txt for details.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Number of threads is empirical: We need enough (>4) threads so that
// the reduction is really performed hierarchically in the barrier!

// RUN: env OMP_NUM_THREADS=3 %libarcher-compile-and-run| FileCheck %s
// RUN: env OMP_NUM_THREADS=7 %libarcher-compile-and-run| FileCheck %s

// REQUIRES: tsan
#include <omp.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  int var = 0;

#pragma omp parallel
  {
#pragma omp masked
    var = 23;
#pragma omp barrier
#pragma omp for reduction(+ : var)
    for (int i = 0; i < 100; i++) {
      var++;
    }
#pragma omp masked
    var += 42;
  }
  fprintf(stderr, "DONE\n");
  int error = (var != 23 + 100 + 42);
  return error;
}

// CHECK-NOT: ThreadSanitizer: data race
// CHECK-NOT: ThreadSanitizer: reported
// CHECK: DONE
