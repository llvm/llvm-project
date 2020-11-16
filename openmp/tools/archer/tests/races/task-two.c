/*
 * task-two.c -- Archer testcase
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
#include <omp.h>
#include <stdio.h>
#include <unistd.h>

#define NUM_THREADS 2

int main(int argc, char *argv[]) {
  int var = 0;
  int i;

#pragma omp parallel for num_threads(NUM_THREADS) shared(var)                  \
    schedule(static, 1)
  for (i = 0; i < NUM_THREADS; i++) {
#pragma omp task shared(var) if (0) // the task is inlined an executed locally
    { var++; }
  }

  int error = (var != 2);
  fprintf(stderr, "DONE\n");
  return error;
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK-NEXT:   {{(Write|Read)}} of size 4
// CHECK-NEXT: #0 {{.*}}task-two.c:30
// CHECK:   Previous write of size 4
// CHECK-NEXT: #0 {{.*}}task-two.c:30
// CHECK: DONE
// CHECK: ThreadSanitizer: reported 1 warnings
