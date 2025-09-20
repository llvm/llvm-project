// This test checks that #pragma omp target update from(data1[0:3:4],
// data2[0:2:5]) correctly updates disjoint strided sections of multiple arrays
// from the device to the host.

// RUN: %libomptarget-compile-run-and-check-generic
#include <omp.h>
#include <stdio.h>

int main() {
  int len = 12;
  double data1[len], data2[len];

// Initial values
#pragma omp target map(tofrom : data1[0 : len], data2[0 : len])
  {
    for (int i = 0; i < len; i++) {
      data1[i] = i;
      data2[i] = i * 10;
    }
  }

  printf("original host array values:\n");
  printf("data1: ");
  for (int i = 0; i < len; i++)
    printf("%.1f ", data1[i]);
  printf("\ndata2: ");
  for (int i = 0; i < len; i++)
    printf("%.1f ", data2[i]);
  printf("\n\n");

#pragma omp target data map(to : data1[0 : len], data2[0 : len])
  {
// Modify arrays on device
#pragma omp target
    {
      for (int i = 0; i < len; i++)
        data1[i] += i;
      for (int i = 0; i < len; i++)
        data2[i] += 100;
    }

// data1[0:3:4]  // indices 0,4,8
// data2[0:2:5]  // indices 0,5
#pragma omp target update from(data1[0 : 3 : 4], data2[0 : 2 : 5])
  }

  printf("device array values after update from:\n");
  printf("data1: ");
  for (int i = 0; i < len; i++)
    printf("%.1f ", data1[i]);
  printf("\ndata2: ");
  for (int i = 0; i < len; i++)
    printf("%.1f ", data2[i]);
  printf("\n\n");

  // CHECK: data1: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0
  // CHECK: data2: 0.0 10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0 90.0 100.0 110.0

  // CHECK: data1: 0.0 1.0 2.0 3.0 8.0 5.0 6.0 7.0 16.0 9.0 10.0 11.0
  // CHECK: data2: 100.0 10.0 20.0 30.0 40.0 150.0 60.0 70.0 80.0 90.0 100.0
  // 110.0
}
