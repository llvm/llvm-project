// XFAIL: amdgcn-amd-amdhsa
// Basic offloading test for function compiled with flang
// REQUIRES: flang, amdgcn-amd-amdhsa

// RUN: %flang -c -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
// RUN:   %S/../../Inputs/basic_array.f90 -o basic_array.o
// RUN: %libomptarget-compile-generic basic_array.o
// RUN: %t | %fcheck-generic

#include <stdio.h>
#define TEST_ARR_LEN 10

#pragma omp declare target
void increment_at(int i, int *array);
#pragma omp end declare target

void increment_array(int *b, int n) {
#pragma omp target map(tofrom : b [0:n])
  for (int i = 0; i < n; i++) {
    increment_at(i, b);
  }
}

int main() {
  int arr[TEST_ARR_LEN] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  increment_array(arr, TEST_ARR_LEN);
  for (int i = 0; i < TEST_ARR_LEN; i++) {
    printf("%d = %d\n", i, arr[i]);
  }

  return 0;
}

// CHECK: 0 = 1
// CHECK-NEXT: 1 = 2
// CHECK-NEXT: 2 = 3
// CHECK-NEXT: 3 = 4
// CHECK-NEXT: 4 = 5
// CHECK-NEXT: 5 = 6
// CHECK-NEXT: 6 = 7
// CHECK-NEXT: 7 = 8
// CHECK-NEXT: 8 = 9
// CHECK-NEXT: 9 = 10
