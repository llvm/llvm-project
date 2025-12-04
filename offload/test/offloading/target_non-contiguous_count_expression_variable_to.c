// RUN: %libomptarget-compile-run-and-check-generic
#include <omp.h>
#include <stdio.h>

int main() {
  int len = 16;
  double data[len];
  double data1[len], data2[len];

  // Initialize data, data1, data2 on device (and copy back to host)
#pragma omp target map(tofrom : len, data[0 : len], data1[0 : len], data2[0 : len])
  {
    for (int i = 0; i < len; i++) {
      data[i] = i;
      data1[i] = i;
      data2[i] = i * 10;
    }
  }

#pragma omp target data map(to : len, data[0 : len], data1[0 : len], data2[0 : len])
  {
    int count = 4;
    // indices: {0, 2, 4, 6}
#pragma omp target update to(data[0 : count : 2])

    int half_len = len / 2;
    // indices: {0, 2, 4, 6, 8, 10, 12, 14}
#pragma omp target update to(data[0 : half_len : 2])

    int dyn_count = len - 4;
    // indices: {2, 4, 6, 8, 10, 12, 14}
#pragma omp target update to(data[2 : dyn_count : 2])

    int partial_count = 4;
    // indices: {0, 3, 6, 9}
#pragma omp target update to(data[0 : partial_count : 3])

    int count1 = 3;
    int count2 = 2;
    // data1 indices: {0, 4, 8}
    // data2 indices: {0, 5}
#pragma omp target update to(data1[0 : count1 : 4], data2[0 : count2 : 5])

    // Device modifies arrays
#pragma omp target
    {
      for (int i = 0; i < len; i++) {
        data[i] += i;    // becomes 2*i on device
        data1[i] += i;   // becomes 2*i on device
        data2[i] += 100; // becomes i*10 + 100 on device
      }
    }
  }

  // Print results
  printf("from target array results (data):\n");
  for (int i = 0; i < len; i++)
    printf("%f\n", data[i]);
  printf("\n");

  printf("from target array results (data1, data2):\n");
  printf("data1:\n");
  for (int i = 0; i < len; i++)
    printf("%f\n", data1[i]);
  printf("data2:\n");
  for (int i = 0; i < len; i++)
    printf("%f\n", data2[i]);
  printf("\n");

  return 0;
}

// CHECK: from target array results (data):
// CHECK: 0.000000
// CHECK: 1.000000
// CHECK: 2.000000
// CHECK: 3.000000
// CHECK: 4.000000
// CHECK: 5.000000
// CHECK: 6.000000
// CHECK: 7.000000
// CHECK: 8.000000
// CHECK: 9.000000
// CHECK: 10.000000
// CHECK: 11.000000
// CHECK: 12.000000
// CHECK: 13.000000
// CHECK: 14.000000
// CHECK: 15.000000

// CHECK: from target array results (data1, data2):
// CHECK: data1:
// CHECK: 0.000000
// CHECK: 1.000000
// CHECK: 2.000000
// CHECK: 3.000000
// CHECK: 4.000000
// CHECK: 5.000000
// CHECK: 6.000000
// CHECK: 7.000000
// CHECK: 8.000000
// CHECK: 9.000000
// CHECK: 10.000000
// CHECK: 11.000000
// CHECK: 12.000000
// CHECK: 13.000000
// CHECK: 14.000000
// CHECK: 15.000000
// CHECK: data2:
// CHECK: 0.000000
// CHECK: 10.000000
// CHECK: 20.000000
// CHECK: 30.000000
// CHECK: 40.000000
// CHECK: 50.000000
// CHECK: 60.000000
// CHECK: 70.000000
// CHECK: 80.000000
// CHECK: 90.000000
// CHECK: 100.000000
// CHECK: 110.000000
// CHECK: 120.000000
// CHECK: 130.000000
// CHECK: 140.000000
// CHECK: 150.000000
