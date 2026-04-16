// RUN: %libomptarget-compile-run-and-check-generic
// Tests multiple arrays with different variable strides in single update
// clause.

#include <omp.h>
#include <stdio.h>

void test_1_update_from_multiple() {
  int stride1 = 2;
  int stride2 = 2;
  double data1[10], data2[10];

  // Initialize data on host
  for (int i = 0; i < 10; i++) {
    data1[i] = i;
    data2[i] = i * 10;
  }

  printf("Test 1: Update FROM - Multiple arrays\n");

#pragma omp target data map(to : stride1, stride2, data1[0 : 10], data2[0 : 10])
  {
#pragma omp target
    {
      for (int i = 0; i < 10; i++) {
        data1[i] += i;
        data2[i] += 100;
      }
    }

#pragma omp target update from(data1[0 : 5 : stride1], data2[0 : 5 : stride2])
  }

  printf("from target data1:\n");
  for (int i = 0; i < 10; i++)
    printf("%f\n", data1[i]);

  printf("\nfrom target data2:\n");
  for (int i = 0; i < 10; i++)
    printf("%f\n", data2[i]);
}

void test_2_update_to_multiple() {
  int stride1 = 2;
  int stride2 = 2;
  double data1[10], data2[10];

  for (int i = 0; i < 10; i++) {
    data1[i] = i;
    data2[i] = i * 10;
  }

  printf("\nTest 2: Update TO - Multiple arrays\n");

#pragma omp target data map(tofrom : stride1, stride2, data1[0 : 10],          \
                                data2[0 : 10])
  {
#pragma omp target
    {
      for (int i = 0; i < 10; i++) {
        data1[i] = 100.0;
        data2[i] = 20.0;
      }
    }

    for (int i = 0; i < 10; i += 2) {
      data1[i] = 10.0;
      data2[i] = 5.0;
    }

#pragma omp target update to(data1[0 : 5 : stride1], data2[0 : 5 : stride2])

#pragma omp target
    {
      for (int i = 0; i < 10; i++) {
        data1[i] += 2.0;
        data2[i] += 2.0;
      }
    }
  }

  printf("device data1 after update to:\n");
  for (int i = 0; i < 10; i++)
    printf("%f\n", data1[i]);

  printf("\ndevice data2 after update to:\n");
  for (int i = 0; i < 10; i++)
    printf("%f\n", data2[i]);
}

int main() {
  test_1_update_from_multiple();
  test_2_update_to_multiple();
  return 0;
}

// CHECK: Test 1: Update FROM - Multiple arrays
// CHECK: from target data1:
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

// CHECK: from target data2:
// CHECK-NEXT: 100.000000
// CHECK-NEXT: 10.000000
// CHECK-NEXT: 120.000000
// CHECK-NEXT: 30.000000
// CHECK-NEXT: 140.000000
// CHECK-NEXT: 50.000000
// CHECK-NEXT: 160.000000
// CHECK-NEXT: 70.000000
// CHECK-NEXT: 180.000000
// CHECK-NEXT: 90.000000

// CHECK: Test 2: Update TO - Multiple arrays
// CHECK: device data1 after update to:
// CHECK-NEXT: 12.000000
// CHECK-NEXT: 102.000000
// CHECK-NEXT: 12.000000
// CHECK-NEXT: 102.000000
// CHECK-NEXT: 12.000000
// CHECK-NEXT: 102.000000
// CHECK-NEXT: 12.000000
// CHECK-NEXT: 102.000000
// CHECK-NEXT: 12.000000
// CHECK-NEXT: 102.000000

// CHECK: device data2 after update to:
// CHECK-NEXT: 7.000000
// CHECK-NEXT: 22.000000
// CHECK-NEXT: 7.000000
// CHECK-NEXT: 22.000000
// CHECK-NEXT: 7.000000
// CHECK-NEXT: 22.000000
// CHECK-NEXT: 7.000000
// CHECK-NEXT: 22.000000
// CHECK-NEXT: 7.000000
// CHECK-NEXT: 22.000000
