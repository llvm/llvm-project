/// Test that llvm-cov supports gcov 8 compatible format.
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
/// FIXME several lines do not match gcov 8

// RUN: rm -rf %t && mkdir %t && cd %t
// RUN: cp %s %p/Inputs/gcov-8.gc* .

/// FIXME Lines executed:100.00% of 12
// RUN: llvm-cov gcov gcov-8.c | FileCheck %s --check-prefixes=OUT,OUTFILE
// OUT:          File 'gcov-8.c'
// OUT-NEXT:     Lines executed:77.78% of 9
// OUT-B-NEXT:   Branches executed:85.71% of 14
// OUT-B-NEXT:   Taken at least once:42.86% of 14
// OUT-B-NEXT:   No calls
// OUTFILE-NEXT: Creating 'gcov-8.c.gcov'
// OUT-EMPTY:

// RUN: FileCheck --input-file=%t/gcov-8.c.gcov --check-prefix=HEADER %s
// RUN: FileCheck --input-file=%t/gcov-8.c.gcov --check-prefix=GCOV %s

// HEADER: {{^}} -:    0:Source:gcov-8.c
// HEADER-NEXT:  -:    0:Graph:gcov-8.gcno
// HEADER-NEXT:  -:    0:Data:gcov-8.gcda
// HEADER-NEXT:  -:    0:Runs:1{{$}}
// HEADER-NEXT:  -:    0:Programs:1
// HEADER-NEXT:  -:    1:/// Test that llvm-cov

// RUN: llvm-cov gcov -i gcov-8.c | FileCheck %s --check-prefix=OUT
// RUN: FileCheck %s --check-prefix=I < gcov-8.c.gcov
// RUN: llvm-cov gcov --intermediate-format gcov-8.c
// RUN: FileCheck %s --check-prefix=I < gcov-8.c.gcov

// RUN: llvm-cov gcov -i -b gcov-8.c | FileCheck %s --check-prefixes=OUT,OUT-B
// RUN: FileCheck %s --check-prefixes=I,I-B < gcov-8.c.gcov

//        I:file:gcov-8.c
//   I-NEXT:function:4,1,main
//   I-NEXT:lcount:4,1
//   I-NEXT:lcount:6,12
// I-B-NEXT:branch:6,taken
// I-B-NEXT:branch:6,nottaken
//   I-NEXT:lcount:7,11
// I-B-NEXT:branch:7,taken
// I-B-NEXT:branch:7,nottaken
//   I-NEXT:lcount:8,7
// I-B-NEXT:branch:8,taken
// I-B-NEXT:branch:8,nottaken
//   I-NEXT:lcount:9,11
//   I-NEXT:lcount:10,11
// I-B-NEXT:branch:10,taken
// I-B-NEXT:branch:10,nottaken
//   I-NEXT:lcount:11,11
// I-B-NEXT:branch:11,taken
// I-B-NEXT:branch:11,nottaken
// I-B-NEXT:branch:11,taken
// I-B-NEXT:branch:11,nottaken
//   I-NEXT:lcount:12,0
// I-B-NEXT:branch:12,notexec
// I-B-NEXT:branch:12,notexec
//   I-NEXT:lcount:14,0
