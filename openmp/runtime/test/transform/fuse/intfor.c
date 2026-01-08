// RUN: %libomp-compile-and-run  | FileCheck %s --match-full-lines

#ifndef HEADER
#define HEADER

#include <stdlib.h>
#include <stdio.h>

int main() {
  printf("do\n");
#pragma omp fuse
  {
    for (int i = 5; i <= 25; i += 5)
      printf("i=%d\n", i);
    for (int j = 10; j < 100; j += 10)
      printf("j=%d\n", j);
    for (int k = 10; k > 0; --k)
      printf("k=%d\n", k);
  }
  printf("done\n");
  return EXIT_SUCCESS;
}
#endif /* HEADER */

// CHECK: do
// CHECK-NEXT: i=5
// CHECK-NEXT: j=10
// CHECK-NEXT: k=10
// CHECK-NEXT: i=10
// CHECK-NEXT: j=20
// CHECK-NEXT: k=9
// CHECK-NEXT: i=15
// CHECK-NEXT: j=30
// CHECK-NEXT: k=8
// CHECK-NEXT: i=20
// CHECK-NEXT: j=40
// CHECK-NEXT: k=7
// CHECK-NEXT: i=25
// CHECK-NEXT: j=50
// CHECK-NEXT: k=6
// CHECK-NEXT: j=60
// CHECK-NEXT: k=5
// CHECK-NEXT: j=70
// CHECK-NEXT: k=4
// CHECK-NEXT: j=80
// CHECK-NEXT: k=3
// CHECK-NEXT: j=90
// CHECK-NEXT: k=2
// CHECK-NEXT: k=1
// CHECK-NEXT: done
