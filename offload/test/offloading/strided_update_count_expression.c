// This test checks that "update from" and "update to" clauses in OpenMP are
// supported when elements are updated in a non-contiguous manner with variable
// count expression. Tests #pragma omp target update from/to(data[0:len/2:2])
// where the count (len/2) is a variable expression, not a constant.

// RUN: %libomptarget-compile-run-and-check-generic
#include <omp.h>
#include <stdio.h>

void test_1_update_from() {
  int len = 10;
  double data[len];

  // Initialize data on host
  for (int i = 0; i < len; i++) {
    data[i] = i;
  }

  printf("Test 1: Update FROM device\n");
  printf("original host array values:\n");
  for (int i = 0; i < len; i++)
    printf("%f\n", data[i]);

#pragma omp target data map(to : len, data[0 : len])
  {
#pragma omp target
    for (int i = 0; i < len; i++) {
      data[i] += i;
    }

#pragma omp target update from(data[0 : len / 2 : 2])
  }

  printf("from target array results:\n");
  for (int i = 0; i < len; i++)
    printf("%f\n", data[i]);
}

void test_2_update_to() {
  int len = 10;
  double data[len];

  for (int i = 0; i < len; i++) {
    data[i] = i;
  }

  printf("\nTest 2: Update TO device\n");
  printf("original host array values:\n");
  for (int i = 0; i < len; i++)
    printf("%f\n", data[i]);

#pragma omp target data map(tofrom : len, data[0 : len])
  {
#pragma omp target
    for (int i = 0; i < len; i++) {
      data[i] = 20.0;
    }

    data[0] = 10.0;
    data[2] = 10.0;
    data[4] = 10.0;
    data[6] = 10.0;
    data[8] = 10.0;

#pragma omp target update to(data[0 : len / 2 : 2])

#pragma omp target
    for (int i = 0; i < len; i++) {
      data[i] += 5.0;
    }
  }

  printf("device array values after update to:\n");
  for (int i = 0; i < len; i++)
    printf("%f\n", data[i]);
}

int main() {
  test_1_update_from();
  test_2_update_to();
  return 0;
}

// CHECK: Test 1: Update FROM device
// CHECK: original host array values:
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

// CHECK: from target array results:
// CHECK-NEXT: 0.000000
// CHECK-NEXT: 1.000000
// CHECK-NEXT: 4.000000
// CHECK-NEXT: 3.000000
// CHECK-NEXT: 8.000000
// CHECK-NEXT: 5.000000
// CHECK-NEXT: 12.000000
// CHECK-NEXT: 7.000000
// CHECK-NEXT: 16.000000
// CHECK-NEXT: 9.000000

// CHECK: Test 2: Update TO device
// CHECK: original host array values:
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

// CHECK: device array values after update to:
// CHECK-NEXT: 15.000000
// CHECK-NEXT: 25.000000
// CHECK-NEXT: 15.000000
// CHECK-NEXT: 25.000000
// CHECK-NEXT: 15.000000
// CHECK-NEXT: 25.000000
// CHECK-NEXT: 15.000000
// CHECK-NEXT: 25.000000
// CHECK-NEXT: 15.000000
// CHECK-NEXT: 25.000000
