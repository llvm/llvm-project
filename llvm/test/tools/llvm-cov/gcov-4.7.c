/// Test that llvm-cov supports gcov [4.7,8) compatible format.
#include <math.h>
#include <stdio.h>
int main() {                                      // GCOV:      #####: [[@LINE]]:int main
  double a[11], result;                           // GCOV-NEXT: -: [[@LINE]]:
  for (int i = 0; i < 11; i++)                    // GCOV-NEXT: #####: [[@LINE]]:
    scanf("%lf", &a[i]);                          // GCOV-NEXT: 11: [[@LINE]]:
  for (int i = 10; i >= 0; i--) {                 // GCOV-NEXT: 4: [[@LINE]]:
    result = sqrt(fabs(a[i])) + 5 * pow(a[i], 3); // GCOV-NEXT: 11: [[@LINE]]:
    printf("\nf(%lf) = ");                        // GCOV-NEXT: 11: [[@LINE]]:
    if (result > 400) printf("Overflow!");        // GCOV-NEXT: #####: [[@LINE]]:
    else printf("%lf", result);                   // GCOV-NEXT: 4: [[@LINE]]:
  }                                               // GCOV-NEXT: -: [[@LINE]]:
  return 0;                                       // GCOV-NEXT: #####: [[@LINE]]:
}                                                 // GCOV-NEXT: -: [[@LINE]]:
/// FIXME several lines do not match gcov 7

// RUN: rm -rf %t && mkdir %t && cd %t
// RUN: cp %s %p/Inputs/gcov-4.7.gc* .

/// FIXME Lines executed:100.00% of 12
// RUN: llvm-cov gcov gcov-4.7.c | FileCheck %s
// CHECK:      File 'gcov-4.7.c'
// CHECK-NEXT: Lines executed:55.56% of 9
// CHECK-NEXT: Creating 'gcov-4.7.c.gcov'

// RUN: FileCheck --input-file=%t/gcov-4.7.c.gcov --check-prefix=HEADER %s
// RUN: FileCheck --input-file=%t/gcov-4.7.c.gcov --check-prefix=GCOV %s

// HEADER: {{^}} -:    0:Source:gcov-4.7.c
// HEADER-NEXT:  -:    0:Graph:gcov-4.7.gcno
// HEADER-NEXT:  -:    0:Data:gcov-4.7.gcda
// HEADER-NEXT:  -:    0:Runs:1{{$}}
// HEADER-NEXT:  -:    0:Programs:1
// HEADER-NEXT:  -:    1:/// Test that llvm-cov
