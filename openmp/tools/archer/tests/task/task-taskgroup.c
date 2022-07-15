// REQUIRES: garbage
/*
 * task-taskgroup.c -- Archer testcase
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
//
// See tools/archer/LICENSE.txt for details.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
// Notified per clause 4(b) of the license.
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
#pragma omp taskgroup
    {
#pragma omp task shared(var, a)
      {
        var++;
        OMPT_SIGNAL(a);
      }

      // Give other thread time to steal the task.
      OMPT_WAIT(a, 1);
    }

    var++;
  }

  fprintf(stderr, "DONE\n");
  int error = (var != 2);
  return error;
}

// CHECK-NOT: ThreadSanitizer: data race
// CHECK-NOT: ThreadSanitizer: reported
// CHECK: DONE
