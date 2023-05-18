/*
 * taskwait-depend.c -- Archer testcase
 * derived from DRB166-taskdep4-orig-omp50-no.c in DataRaceBench
 */
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
//
// See tools/archer/LICENSE.txt for details.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %libarcher-compile-and-run | FileCheck %s
// REQUIRES: tsan

#include "ompt/ompt-signal.h"
#include <omp.h>
#include <stdio.h>

void foo() {

  int x = 0, y = 2, sem = 0;

#pragma omp task depend(inout : x) shared(x, sem)
  {
    OMPT_SIGNAL(sem);
    x++; // 1st Child Task
  }

#pragma omp task shared(y, sem)
  {
    OMPT_SIGNAL(sem);
    y--; // 2nd child task
  }

  OMPT_WAIT(sem, 2);
#pragma omp taskwait depend(in : x) // 1st taskwait

  printf("x=%d\n", x);

#pragma omp taskwait // 2nd taskwait

  printf("y=%d\n", y);
}

int main() {
#pragma omp parallel num_threads(2)
#pragma omp single
  foo();

  return 0;
}

// CHECK-NOT: ThreadSanitizer: data race
// CHECK-NOT: ThreadSanitizer: reported
// CHECK: y=1
