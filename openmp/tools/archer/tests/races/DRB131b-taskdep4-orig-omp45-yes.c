/*
 * DRB131b-taskdep4-orig-omp45-yes.c -- Archer testcase
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
#include <omp.h>
#include <stdio.h>

int sem = 0;

void foo() {

  int x = 0, y = 2;

#pragma omp task depend(inout : x) shared(x, sem)
  {
    OMPT_SIGNAL(sem);
    x++; //1st Child Task
  }

#pragma omp task shared(y, sem)
  {
    OMPT_SIGNAL(sem);
    y--; // 2nd child task
  }

#pragma omp task depend(in : x) if (0) // 1st taskwait
  {}

  printf("x=%d\n", x);
  printf("y=%d\n", y);
#pragma omp taskwait // 2nd taskwait
}

int main() {
#pragma omp parallel
  {
#pragma omp masked
    foo();
    OMPT_WAIT(sem, 2);
  }

  return 0;
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK: ThreadSanitizer: reported {{[0-9]+}} warnings
