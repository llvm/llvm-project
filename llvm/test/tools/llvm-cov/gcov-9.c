/// Test that llvm-cov supports gcov 9 compatible format.
#include <math.h>
#include <stdio.h>
int main() {                                      // GCOV:      1: [[@LINE]]:int main
  double a[11], result;                           // GCOV-NEXT: -: [[@LINE]]:
  for (int i = 0; i < 11; i++)                    // GCOV-NEXT: 12: [[@LINE]]:
    scanf("%lf", &a[i]);                          // GCOV-NEXT: 11: [[@LINE]]:
  for (int i = 10; i >= 0; i--) {                 // GCOV-NEXT: 7: [[@LINE]]:
    result = sqrt(fabs(a[i])) + 5 * pow(a[i], 3); // GCOV-NEXT: 11: [[@LINE]]:
    printf("\nf(%lf) = ");                        // GCOV-NEXT: 11: [[@LINE]]:
    if (result > 400) printf("Overflow!");        // GCOV-NEXT: 11: [[@LINE]]:
    else printf("%lf", result);                   // GCOV-NEXT: #####: [[@LINE]]:
  }                                               // GCOV-NEXT: -: [[@LINE]]:
  return 0;                                       // GCOV-NEXT: #####: [[@LINE]]:
}                                                 // GCOV-NEXT: -: [[@LINE]]:
/// FIXME several lines do not match gcov 9

// RUN: rm -rf %t && mkdir %t && cd %t
// RUN: cp %s %p/Inputs/gcov-9.gc* .

/// FIXME Lines executed:100.00% of 12
// RUN: llvm-cov gcov gcov-9.c | FileCheck %s
// CHECK:      File 'gcov-9.c'
// CHECK-NEXT: Lines executed:77.78% of 9
// CHECK-NEXT: gcov-9.c:creating 'gcov-9.c.gcov'

// RUN: FileCheck --input-file=%t/gcov-9.c.gcov --check-prefix=HEADER %s
// RUN: FileCheck --input-file=%t/gcov-9.c.gcov --check-prefix=GCOV %s

// HEADER: {{^}} -:    0:Source:gcov-9.c
// HEADER-NEXT:  -:    0:Graph:gcov-9.gcno
// HEADER-NEXT:  -:    0:Data:gcov-9.gcda
// HEADER-NEXT:  -:    0:Runs:1{{$}}
// HEADER-NEXT:  -:    1:/// Test that llvm-cov
