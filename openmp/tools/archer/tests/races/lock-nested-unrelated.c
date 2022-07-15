// REQUIRES: garbage
/*
 * lock-nested-unrelated.c -- Archer testcase
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

// RUN: %libarcher-compile-and-run-race | FileCheck %s
// RUN: %libarcher-compile-and-run-race-noserial | FileCheck %s
// REQUIRES: tsan
#include <omp.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  int var = 0;

  omp_nest_lock_t lock;
  omp_init_nest_lock(&lock);

#pragma omp parallel num_threads(8) shared(var)
  {
    omp_set_nest_lock(&lock);
    omp_set_nest_lock(&lock);
    // Dummy locking.
    omp_unset_nest_lock(&lock);
    omp_unset_nest_lock(&lock);

    var++;
  }

  omp_destroy_nest_lock(&lock);

  fprintf(stderr, "DONE\n");
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK-NEXT:   {{(Write|Read)}} of size 4
// CHECK-NEXT: #0 {{.*}}lock-nested-unrelated.c:33
// CHECK:   Previous write of size 4
// CHECK-NEXT: #0 {{.*}}lock-nested-unrelated.c:33
// CHECK: DONE
// CHECK: ThreadSanitizer: reported 1 warnings
