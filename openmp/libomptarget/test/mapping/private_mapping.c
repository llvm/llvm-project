// RUN: %libomptarget-compile-run-and-check-generic

#include <assert.h>
#include <stdio.h>

int main() {
  int data1[3] = {1, 2, 5};
  int data2[3] = {10, 20, 50};
  int data3[3] = {100, 200, 500};
  int sum[16] = {0};

  for (int i=0; i<16; i++) sum[i] = 10000;

#pragma omp target teams distribute parallel for map(tofrom : sum)             \
    firstprivate(data1, data2, data3)
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 3; ++j) {
      sum[i] += data1[j];
      sum[i] += data2[j];
      sum[i] += data3[j];
    }
  }

  int correct = 1;
  for (int i = 0; i < 16; ++i) {
    if (sum[i] != 10888) {
      correct = 0;
      printf("ERROR: The sum for index %d is %d\n", i, sum[i]);
      printf("ERROR: data1 = {%d, %d, %d}\n", data1[0], data1[1], data1[2]);
      printf("ERROR: data2 = {%d, %d, %d}\n", data2[0], data2[1], data2[2]);
      printf("ERROR: data3 = {%d, %d, %d}\n", data3[0], data3[1], data3[2]);
      break;
    }
  }
  fflush(stdout);
  assert(correct);

  printf("PASS\n");

  return 0;
}

// CHECK: PASS
