// RUN: %libomptarget-compile-run-and-check-generic
// Miscellaneous variable stride tests: stride=1, stride=array_size, stride from
// array subscript.

#include <omp.h>
#include <stdio.h>

void test_1_variable_stride_one() {
  int stride_one = 1;
  double data1[10];

  // Initialize data on host
  for (int i = 0; i < 10; i++) {
    data1[i] = i;
  }

#pragma omp target data map(to : stride_one, data1[0 : 10])
  {
#pragma omp target
    {
      for (int i = 0; i < 10; i++) {
        data1[i] += i;
      }
    }

#pragma omp target update from(data1[0 : 10 : stride_one])
  }

  printf("Test 1: Variable stride = 1\n");
  for (int i = 0; i < 10; i++)
    printf("%f\n", data1[i]);
}

void test_2_variable_stride_large() {
  int stride_large = 5;
  double data2[10];

  // Initialize data on host
  for (int i = 0; i < 10; i++) {
    data2[i] = i;
  }

#pragma omp target data map(to : stride_large, data2[0 : 10])
  {
#pragma omp target
    {
      for (int i = 0; i < 10; i++) {
        data2[i] += i;
      }
    }

#pragma omp target update from(data2[0 : 2 : stride_large])
  }

  printf("\nTest 2: Variable stride = 5\n");
  for (int i = 0; i < 10; i++)
    printf("%f\n", data2[i]);
}

int main() {
  test_1_variable_stride_one();
  test_2_variable_stride_large();
  return 0;
}

// CHECK: Test 1: Variable stride = 1
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

// CHECK: Test 2: Variable stride = 5
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
