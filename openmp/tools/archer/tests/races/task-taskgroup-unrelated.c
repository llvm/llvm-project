/*
 * task-taskgroup-unrelated.c -- Archer testcase
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
#pragma omp task shared(var, a)
    {
      var++;
      OMPT_SIGNAL(a);
      // Give master thread time to execute the task in the taskgroup.
      OMPT_WAIT(a, 2);
    }

#pragma omp taskgroup
    {
#pragma omp task if (0)
      {
        // Dummy task.
      }

      // Give other threads time to steal the tasks.
      OMPT_WAIT(a, 1);
      OMPT_SIGNAL(a);
    }

    var++;
  }

  int error = (var != 2);
  fprintf(stderr, "DONE\n");
  return error;
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK-NEXT:   {{(Write|Read)}} of size 4
// CHECK-NEXT: #0 {{.*}}task-taskgroup-unrelated.c:46
// CHECK:   Previous write of size 4
// CHECK-NEXT: #0 {{.*}}task-taskgroup-unrelated.c:28
// CHECK: DONE
// CHECK: ThreadSanitizer: reported 1 warnings

