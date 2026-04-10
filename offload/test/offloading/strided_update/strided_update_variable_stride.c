// This test checks "update from" and "update to" with variable stride.
// Tests data[0:5:stride] where stride is a variable, making it non-contiguous.

// RUN: %libomptarget-compile-run-and-check-generic
#include <omp.h>
#include <stdio.h>

void test_1_update_from() {
  int stride = 2;
  double data[10];

  // Initialize data on host
  for (int i = 0; i < 10; i++) {
    data[i] = i;
  }

  printf("Test 1: Update FROM device\n");
  printf("original values:\n");
  for (int i = 0; i < 10; i++)
    printf("%f\n", data[i]);

#pragma omp target data map(to : stride, data[0 : 10])
  {
#pragma omp target
    {
      for (int i = 0; i < 10; i++) {
        data[i] += i;
      }
    }

#pragma omp target update from(data[0 : 5 : stride])
  }

  printf("from target results:\n");
  for (int i = 0; i < 10; i++)
    printf("%f\n", data[i]);
}

void test_2_update_to() {
  int stride = 2;
  double data[10];

  for (int i = 0; i < 10; i++) {
    data[i] = i;
  }

  printf("\nTest 2: Update TO device\n");
  printf("original values:\n");
  for (int i = 0; i < 10; i++)
    printf("%f\n", data[i]);

#pragma omp target data map(tofrom : stride, data[0 : 10])
  {
#pragma omp target
    {
      for (int i = 0; i < 10; i++) {
        data[i] = 50.0;
      }
    }

    for (int i = 0; i < 10; i += 2) {
      data[i] = 10.0;
    }

#pragma omp target update to(data[0 : 5 : stride])

#pragma omp target
    {
      for (int i = 0; i < 10; i++) {
        data[i] += 5.0;
      }
    }
  }

  printf("device values after update to:\n");
  for (int i = 0; i < 10; i++)
    printf("%f\n", data[i]);
}

int main() {
  test_1_update_from();
  test_2_update_to();
  return 0;
}

// CHECK: Test 1: Update FROM device
// CHECK: original values:
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

// CHECK: from target results:
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
// CHECK: original values:
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

// CHECK: device values after update to:
// CHECK-NEXT: 15.000000
// CHECK-NEXT: 55.000000
// CHECK-NEXT: 15.000000
// CHECK-NEXT: 55.000000
// CHECK-NEXT: 15.000000
// CHECK-NEXT: 55.000000
// CHECK-NEXT: 15.000000
// CHECK-NEXT: 55.000000
// CHECK-NEXT: 15.000000
// CHECK-NEXT: 55.000000
