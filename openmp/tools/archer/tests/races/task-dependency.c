/*
 * task-dependency.c -- Archer testcase
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
#include <unistd.h>

int main(int argc, char *argv[]) {
  int var = 0, a = 0;

#pragma omp parallel num_threads(2) shared(var, a)
#pragma omp master
  {
#pragma omp task shared(var, a) depend(out : var)
    {
      OMPT_SIGNAL(a);
      var++;
    }

#pragma omp task shared(a) depend(in : var)
    {
      OMPT_SIGNAL(a);
      OMPT_WAIT(a, 3);
    }

#pragma omp task shared(var) // depend(in: var) is missing here!
    {
      var++;
      OMPT_SIGNAL(a);
    }

    // Give other thread time to steal the task.
    OMPT_WAIT(a, 2);
  }

  int error = (var != 2);
  fprintf(stderr, "DONE\n");
  return error;
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK-NEXT:   {{(Write|Read)}} of size 4
// CHECK-NEXT: #0 {{.*}}task-dependency.c:41
// CHECK:   Previous write of size 4
// CHECK-NEXT: #0 {{.*}}task-dependency.c:30
// CHECK: DONE
// CHECK: ThreadSanitizer: reported 1 warnings
