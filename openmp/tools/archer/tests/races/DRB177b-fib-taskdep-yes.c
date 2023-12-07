/*
 * DRB177b-fib-taskdep-yes.c -- Archer testcase
 */
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
//
// See tools/archer/LICENSE.txt for details.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %libarcher-compile && env ARCHER_OPTIONS=tasking=1 %libarcher-run-race | FileCheck %s
// RUN: %libarcher-compile && env ARCHER_OPTIONS=tasking=1:ignore_serial=1 %libarcher-run-race | FileCheck %s
// REQUIRES: tsan
#include "ompt/ompt-signal.h"
#include <stdio.h>
#include <stdlib.h>

int sem = 0;

int fib(int n) {
  int i, j, s;
  if (n < 2)
    return n;
#pragma omp task shared(i, sem) depend(out : i)
  { i = fib(n - 1); }
#pragma omp task shared(j, sem) depend(out : j)
  { j = fib(n - 2); }
#pragma omp task shared(i, j, s, sem) depend(in : j)
  { s = i + j; }
#pragma omp taskwait
  return s;
}

int main(int argc, char **argv) {
  int n = 10;
  if (argc > 1)
    n = atoi(argv[1]);
#pragma omp parallel
  {
#pragma omp masked
    {
      printf("fib(%i) = %i\n", n, fib(n));
      OMPT_SIGNAL(sem);
    }
    OMPT_WAIT(sem, 1);
  }
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK: ThreadSanitizer: reported {{[0-9]+}} warnings
