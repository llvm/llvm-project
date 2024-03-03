/*
 * DRB136b-taskdep-mutexinoutset-orig-yes.c -- Archer testcase
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

int main() {
  int a, b, c, d, sem = 0;

#pragma omp parallel num_threads(2)
  {
#pragma omp masked
    {
#pragma omp task depend(out : c)
      {
        OMPT_SIGNAL(sem);
        c = 1;
      }
#pragma omp task depend(out : a)
      {
        OMPT_SIGNAL(sem);
        a = 2;
      }
#pragma omp task depend(out : b)
      {
        OMPT_SIGNAL(sem);
        b = 3;
      }
#pragma omp task depend(in : a)
      {
        OMPT_SIGNAL(sem);
        c += a;
      }
#pragma omp task depend(in : b)
      {
        OMPT_SIGNAL(sem);
        c += b;
      }
#pragma omp task depend(in : c)
      {
        OMPT_SIGNAL(sem);
        d = c;
      }
#pragma omp taskwait {}
    }
    OMPT_WAIT(sem, 6);
  }

  printf("%d\n", d);
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK: ThreadSanitizer: reported {{[0-9]+}} warnings
