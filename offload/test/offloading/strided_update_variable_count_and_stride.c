// RUN: %libomptarget-compile-run-and-check-generic
// Tests combining variable count expression AND variable stride in array
// sections.

#include <omp.h>
#include <stdio.h>

void test_1_update_from() {
  int len = 10;
  int stride = 2;
  double data[len];

  // Initialize data on host
  for (int i = 0; i < len; i++) {
    data[i] = i;
  }

  printf("Test 1: Update FROM - Variable count and stride\n");
  printf("original values:\n");
  for (int i = 0; i < len; i++)
    printf("%f\n", data[i]);

#pragma omp target data map(to : len, stride, data[0 : len])
  {
#pragma omp target
    {
      for (int i = 0; i < len; i++) {
        data[i] += i;
      }
    }

#pragma omp target update from(data[0 : len / 2 : stride])
  }

  printf("from target results:\n");
  for (int i = 0; i < len; i++)
    printf("%f\n", data[i]);
}

void test_2_update_to() {
  int len = 10;
  int stride = 2;
  double data[len];

  for (int i = 0; i < len; i++) {
    data[i] = i;
  }

  printf("\nTest 2: Update TO - Variable count and stride\n");
  printf("original values:\n");
  for (int i = 0; i < len; i++)
    printf("%f\n", data[i]);

#pragma omp target data map(tofrom : len, stride, data[0 : len])
  {
#pragma omp target
    {
      for (int i = 0; i < len; i++) {
        data[i] = 50.0;
      }
    }

    for (int i = 0; i < len / 2; i++) {
      data[i * stride] = 10.0;
    }

#pragma omp target update to(data[0 : len / 2 : stride])

#pragma omp target
    {
      for (int i = 0; i < len; i++) {
        data[i] += 5.0;
      }
    }
  }

  printf("device values after update to:\n");
  for (int i = 0; i < len; i++)
    printf("%f\n", data[i]);
}

int main() {
  test_1_update_from();
  test_2_update_to();
  return 0;
}

// CHECK: Test 1: Update FROM - Variable count and stride
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

// CHECK: Test 2: Update TO - Variable count and stride
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
