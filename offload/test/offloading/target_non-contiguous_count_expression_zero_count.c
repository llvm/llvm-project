
// RUN: %libomptarget-compile-run-and-check-generic
#include <omp.h>
#include <stdio.h>

int main() {
  int len = 8;
  double data[len];
#pragma omp target map(tofrom : len, data[0 : len])
  {
    for (int i = 0; i < len; i++) {
      data[i] = i;
    }
  }

#pragma omp target data map(to : len, data[0 : len])
  {
#pragma omp target
    for (int i = 0; i < len; i++) {
      data[i] += i;
    }

    int zero_count = 0;
#pragma omp target update from(data[0 : zero_count : 2])
  }

  printf("from target array results:\n");
  for (int i = 0; i < len; i++)
    printf("%f\n", data[i]);
  printf("\n");

  return 0;
}

// CHECK: from target array results:
// CHECK: 0.000000
// CHECK: 1.000000
// CHECK: 2.000000
// CHECK: 3.000000
// CHECK: 4.000000
// CHECK: 5.000000
// CHECK: 6.000000
// CHECK: 7.000000
