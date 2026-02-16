// This test checks "update from" and "update to" with multiple arrays and
// variable count expressions. Tests both: (1) multiple arrays in single update
// clause with different count expressions, and (2) overlapping updates to the
// same array with various count expressions.

// RUN: %libomptarget-compile-run-and-check-generic
#include <omp.h>
#include <stdio.h>

void test_1_update_from_multiple() {
  int n1 = 10, n2 = 10;
  double arr1[n1], arr2[n2];

#pragma omp target map(tofrom : n1, n2, arr1[0 : n1], arr2[0 : n2])
  {
    for (int i = 0; i < n1; i++) {
      arr1[i] = i;
    }
    for (int i = 0; i < n2; i++) {
      arr2[i] = i * 10;
    }
  }

  printf("Test 1: Update FROM - Multiple arrays\n");

#pragma omp target data map(to : n1, n2, arr1[0 : n1], arr2[0 : n2])
  {
#pragma omp target
    {
      for (int i = 0; i < n1; i++) {
        arr1[i] += i;
      }
      for (int i = 0; i < n2; i++) {
        arr2[i] += 100;
      }
    }

    // Update with different count expressions in single clause:
    // arr1[0:n1/2:2] = arr1[0:5:2] updates indices 0,2,4,6,8
    // arr2[0:n2/5:2] = arr2[0:2:2] updates indices 0,2
#pragma omp target update from(arr1[0 : n1 / 2 : 2], arr2[0 : n2 / 5 : 2])
  }

  printf("from target arr1 results:\n");
  for (int i = 0; i < n1; i++)
    printf("%f\n", arr1[i]);

  printf("\nfrom target arr2 results:\n");
  for (int i = 0; i < n2; i++)
    printf("%f\n", arr2[i]);
}

void test_2_update_to_multiple() {
  int n1 = 10, n2 = 10;
  double arr1[n1], arr2[n2];

  for (int i = 0; i < n1; i++) {
    arr1[i] = i;
  }
  for (int i = 0; i < n2; i++) {
    arr2[i] = i * 10;
  }

  printf("\nTest 2: Update TO - Multiple arrays\n");

#pragma omp target data map(tofrom : n1, n2, arr1[0 : n1], arr2[0 : n2])
  {
#pragma omp target
    {
      for (int i = 0; i < n1; i++) {
        arr1[i] = 100.0;
      }
      for (int i = 0; i < n2; i++) {
        arr2[i] = 20.0;
      }
    }

    // Modify host
    for (int i = 0; i < n1; i += 2) {
      arr1[i] = 10.0;
    }
    for (int i = 0; i < n2; i += 2) {
      arr2[i] = 5.0;
    }

#pragma omp target update to(arr1[0 : n1 / 2 : 2], arr2[0 : n2 / 5 : 2])

#pragma omp target
    {
      for (int i = 0; i < n1; i++) {
        arr1[i] += 2.0;
      }
      for (int i = 0; i < n2; i++) {
        arr2[i] += 2.0;
      }
    }
  }

  printf("device arr1 values after update to:\n");
  for (int i = 0; i < n1; i++)
    printf("%f\n", arr1[i]);

  printf("\ndevice arr2 values after update to:\n");
  for (int i = 0; i < n2; i++)
    printf("%f\n", arr2[i]);
}

int main() {
  test_1_update_from_multiple();
  test_2_update_to_multiple();
  return 0;
}

// CHECK: Test 1: Update FROM - Multiple arrays
// CHECK: from target arr1 results:
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

// CHECK: from target arr2 results:
// CHECK-NEXT: 100.000000
// CHECK-NEXT: 10.000000
// CHECK-NEXT: 120.000000
// CHECK-NEXT: 30.000000
// CHECK-NEXT: 40.000000
// CHECK-NEXT: 50.000000
// CHECK-NEXT: 60.000000
// CHECK-NEXT: 70.000000
// CHECK-NEXT: 80.000000
// CHECK-NEXT: 90.000000

// CHECK: Test 2: Update TO - Multiple arrays
// CHECK: device arr1 values after update to:
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

// CHECK: device arr2 values after update to:
// CHECK-NEXT: 7.000000
// CHECK-NEXT: 22.000000
// CHECK-NEXT: 7.000000
// CHECK-NEXT: 22.000000
// CHECK-NEXT: 22.000000
// CHECK-NEXT: 22.000000
// CHECK-NEXT: 22.000000
// CHECK-NEXT: 22.000000
// CHECK-NEXT: 22.000000
// CHECK-NEXT: 22.000000
