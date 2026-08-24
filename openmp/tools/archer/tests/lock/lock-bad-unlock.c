/*
 * lock-bad-unlock.c -- Archer testcase
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

#pragma omp parallel num_threads(2)
  {
    if (omp_get_thread_num() == 0)
      omp_set_lock(&lock);
    else
      omp_unset_lock(&lock);
  }

  omp_destroy_lock(&lock);

  fprintf(stderr, "DONE\n");
  return 1;
}

// CHECK: WARNING: ThreadSanitizer: unlock of an unlocked mutex (or by a wrong thread)
// CHECK:     #0 {{.*}}lock-bad-unlock.c:28
// CHECK:   Mutex {{.*}} created at:
// CHECK:     #0 {{.*}}lock-bad-unlock.c:21
// CHECK: SUMMARY: ThreadSanitizer: unlock of an unlocked mutex (or by a wrong thread)
