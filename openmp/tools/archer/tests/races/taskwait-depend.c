/*
 * taskwait-depend.c -- Archer testcase
 * derived from DRB165-taskdep4-orig-omp50-yes.c in DataRaceBench
 */
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
//
// See tools/archer/LICENSE.txt for details.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %libarcher-compile-and-run-race | FileCheck %s
// RUN: %libarcher-compile-and-run-race-noserial | FileCheck %s
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
  printf("y=%d\n", y);
#pragma omp taskwait // 2nd taskwait
}

int main() {
#pragma omp parallel num_threads(2)
#pragma omp single
  foo();

  return 0;
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK-NEXT:   {{(Write|Read)}} of size 4
// CHECK-NEXT: #0 {{.*}}taskwait-depend.c:42:20
// CHECK:   Previous write of size 4
// CHECK-NEXT: #0 {{.*}}taskwait-depend.c:35:6
// CHECK: ThreadSanitizer: reported {{[0-9]+}} warnings
