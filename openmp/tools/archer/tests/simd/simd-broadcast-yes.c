/*
 * simd-broadcast-yes.c -- Archer testcase
 */
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
//
// See tools/LICENSE.txt for details.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %libarcher-compile -DTYPE=float && %libarcher-run-race \
// RUN: | FileCheck --check-prefix=FLOAT %s
// RUN: %libarcher-compile -DTYPE=double && %libarcher-run-race \
// RUN: | FileCheck --check-prefix=DOUBLE %s
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
  TYPE a[len];
  for (int i = 0; i < len; i++)
    a[i] = i;
  double c = M_PI;

#pragma omp parallel for simd num_threads(2) schedule(dynamic, 64)
  for (int i = 0; i < len; i++)
    a[i] = a[i] + a[64];

  fprintf(stderr, "DONE\n");
  return 0;
}

// FLOAT: WARNING: ThreadSanitizer: data race
// FLOAT-NEXT:   {{(Write|Read)}} of size {{(4|8)}}
// FLOAT-NEXT: #0 {{.*}}simd-broadcast-yes.c
// FLOAT:   Previous {{(read|write)}} of size {{(4|8)}}
// FLOAT-NEXT: #0 {{.*}}simd-broadcast-yes.c

// DOUBLE: WARNING: ThreadSanitizer: data race
// DOUBLE-NEXT:   {{(Write|Read)}} of size 8
// DOUBLE-NEXT: #0 {{.*}}simd-broadcast-yes.c
// DOUBLE:   Previous {{(read|write)}} of size 8
// DOUBLE-NEXT: #0 {{.*}}simd-broadcast-yes.c

// CHECK: DONE
// CHECK: ThreadSanitizer: reported {{[0-9]+}} warnings
