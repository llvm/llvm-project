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


// RUN: %libarcher-compile-and-run | FileCheck %s
// REQUIRES: tsan
#include <omp.h>
#include <stdio.h>
#include <unistd.h>
#include "ompt/ompt-signal.h"

int main(int argc, char *argv[]) {
  int var = 0, a = 0;

#pragma omp parallel num_threads(2) shared(var, a)
#pragma omp master
  {
#pragma omp task shared(var, a) depend(out : var)
    {
      var++;
      OMPT_SIGNAL(a);
    }

#pragma omp task shared(var, a) depend(in : var)
    { OMPT_WAIT(a, 2); }

#pragma omp task shared(var, a) depend(in : var)
    {
      OMPT_SIGNAL(a);
      var++;
    }

    // Give other thread time to steal the task.
    OMPT_WAIT(a, 1);
  }

  fprintf(stderr, "DONE\n");
  int error = (var != 2);
  return error;
}

// CHECK-NOT: ThreadSanitizer: data race
// CHECK-NOT: ThreadSanitizer: reported
// CHECK: DONE
