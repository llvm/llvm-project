// RUN: %libomptarget-compile-run-and-check-generic
// Miscellaneous tests for count expressions: tests modulo, large stride with
// computed count, and boundary calculations to ensure expression semantics work
// correctly.

#include <omp.h>
#include <stdio.h>

int main() {
  // ====================================================================
  // TEST 1: Modulo operation in count expression
  // ====================================================================

  int len1 = 10;
  int divisor = 5;
  double data1[len1];

#pragma omp target map(tofrom : len1, divisor, data1[0 : len1])
  {
    for (int i = 0; i < len1; i++) {
      data1[i] = i;
    }
  }

#pragma omp target data map(to : len1, divisor, data1[0 : len1])
  {
#pragma omp target
    {
      for (int i = 0; i < len1; i++) {
        data1[i] += i;
      }
    }

    // data[0:10%5:2] = data[0:0:2] updates no indices (count=0)
#pragma omp target update from(data1[0 : len1 % divisor : 2])
  }

  printf("Test 1: Modulo count expression\n");
  for (int i = 0; i < len1; i++)
    printf("%f\n", data1[i]);

  // ====================================================================
  // TEST 2: Large stride with computed count for boundary coverage
  // ====================================================================

  int len2 = 10;
  int stride = 5;
  double data2[len2];

#pragma omp target map(tofrom : len2, stride, data2[0 : len2])
  {
    for (int i = 0; i < len2; i++) {
      data2[i] = i;
    }
  }

#pragma omp target data map(to : len2, stride, data2[0 : len2])
  {
#pragma omp target
    {
      for (int i = 0; i < len2; i++) {
        data2[i] += i;
      }
    }

    // data[0:(10+5-1)/5:5] = data[0:2:5] updates indices: 0, 5
#pragma omp target update from(data2[0 : (len2 + stride - 1) / stride : stride])
  }

  printf("\nTest 2: Large stride count expression\n");
  for (int i = 0; i < len2; i++)
    printf("%f\n", data2[i]);

  return 0;
}

// CHECK: Test 1: Modulo count expression
// CHECK-NEXT: 0.000000
// CHECK-NEXT: 1.000000
// CHECK-NEXT: 2.000000
// CHECK-NEXT: 3.000000
// CHECK-NEXT: 4.000000
// CHECK-NEXT: 5.000000
// CHECK-NEXT: 6.000000
// CHECK-NEXT: 7.000000
// CHECK-NEXT: 8.000000
// CHECK-NEXT: 9.000000

// CHECK: Test 2: Large stride count expression
// CHECK-NEXT: 0.000000
// CHECK-NEXT: 1.000000
// CHECK-NEXT: 2.000000
// CHECK-NEXT: 3.000000
// CHECK-NEXT: 4.000000
// CHECK-NEXT: 10.000000
// CHECK-NEXT: 6.000000
// CHECK-NEXT: 7.000000
// CHECK-NEXT: 8.000000
// CHECK-NEXT: 9.000000
