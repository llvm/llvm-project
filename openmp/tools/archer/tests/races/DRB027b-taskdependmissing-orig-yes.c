/*
 * DRB027b-taskdependmissing-orig-yes.c -- Archer testcase
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
#include <assert.h>
#include <stdio.h>

int main() {
  int i = 0, sem = 0;
#pragma omp parallel shared(sem) num_threads(2)
  {
#pragma omp masked
    {
#pragma omp task
      {
        OMPT_SIGNAL(sem);
        i = 1;
      }
#pragma omp task
      {
        OMPT_SIGNAL(sem);
        i = 2;
      }
#pragma omp taskwait {}
    }
    OMPT_WAIT(sem, 2);
  }
  printf("i=%d\n", i);
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK: ThreadSanitizer: reported {{[0-9]+}} warnings
