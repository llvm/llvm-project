/*
 * lock-destroy-locked.c -- Archer testcase
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

int main(int argc, char *argv[]) {

  omp_lock_t lock;
  omp_init_lock(&lock);

#pragma omp parallel num_threads(1)
  { omp_set_lock(&lock); }

  omp_destroy_lock(&lock);

  fprintf(stderr, "DONE\n");
  return 1;
}

// CHECK: WARNING: ThreadSanitizer: destroy of a locked mutex
// CHECK:     #0 {{.*}}lock-destroy-locked.c:26
// CHECK:   and:
// CHECK:     #0 {{.*}}lock-destroy-locked.c:24
// CHECK:   Mutex {{.*}} created at:
// CHECK:     #0 {{.*}}lock-destroy-locked.c:21
// CHECK: SUMMARY: ThreadSanitizer: destroy of a locked mutex{{.*}}main
