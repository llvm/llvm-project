/*
 * simd-broadcast-no.c -- Archer testcase
 */
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
//
// See tools/LICENSE.txt for details.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %libarcher-compile -DTYPE=float && %libarcher-run | FileCheck %s
// RUN: %libarcher-compile -DTYPE=double && %libarcher-run | FileCheck %s
// REQUIRES: tsan

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef TYPE
#define TYPE double
#endif /*TYPE*/

int main(int argc, char *argv[]) {
  int len = 20000;
  if (argc > 1)
    len = atoi(argv[1]);
  double a[len];
  for (int i = 0; i < len; i++)
    a[i] = i;
  TYPE c = M_PI;

#pragma omp parallel for simd num_threads(2) schedule(dynamic, 64)
  for (int i = 0; i < len; i++)
    a[i] = a[i] + c;

  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK-NOT: ThreadSanitizer: data race
// CHECK-NOT: ThreadSanitizer: reported
// CHECK: DONE
