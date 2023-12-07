/*
 * DRB173b-non-sibling-taskdep-yes.c -- Archer testcase
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
// XFAIL: *
#include "ompt/ompt-signal.h"
#include <omp.h>
#include <stdio.h>

void foo() {
  int a = 0, sem = 0;

#pragma omp parallel num_threads(2)
  {
#pragma omp masked
#pragma omp taskgroup
    {
#pragma omp task depend(inout : a) shared(a)
      {
#pragma omp task depend(inout : a) shared(a)
        OMPT_SIGNAL(sem);
        a++;
      }

#pragma omp task depend(inout : a) shared(a)
      {
#pragma omp task depend(inout : a) shared(a)
        OMPT_SIGNAL(sem);
        a++;
      }
    }
    OMPT_WAIT(sem, 2);
  }
  printf("a=%d\n", a);
}

int main() {
  foo();

  return 0;
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK: ThreadSanitizer: reported {{[0-9]+}} warnings
