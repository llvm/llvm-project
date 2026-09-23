// RUN: %libomptarget-compile-run-and-check-generic
// Tests non-contiguous array sections with variable stride on struct member
// arrays.

#include <omp.h>
#include <stdio.h>

struct S {
  int stride;
  double data[20];
};

int main() {
  struct S s;
  s.stride = 2;
  int len = 10;

  // Initialize
#pragma omp target map(tofrom : s, len)
  {
    for (int i = 0; i < len; i++) {
      s.data[i] = i;
    }
  }

  // Test FROM
#pragma omp target data map(to : s, len)
  {
#pragma omp target
    {
      for (int i = 0; i < len; i++) {
        s.data[i] += i * 10;
      }
    }

#pragma omp target update from(s.data[0 : 5 : s.stride])
  }

  printf("struct variable stride (from):\n");
  for (int i = 0; i < len; i++)
    printf("%f\n", s.data[i]);

  // Test TO: Reset, modify host, update to device
#pragma omp target map(tofrom : s)
  {
    for (int i = 0; i < len; i++) {
      s.data[i] = i * 2;
    }
  }

  for (int i = 0; i < 5; i++) {
    s.data[i * s.stride] = i + 100;
  }

#pragma omp target data map(to : s)
  {
#pragma omp target update to(s.data[0 : 5 : s.stride])

#pragma omp target
    {
      for (int i = 0; i < len; i++) {
        s.data[i] += 100;
      }
    }
  }

  printf("struct variable stride (to):\n");
  for (int i = 0; i < len; i++)
    printf("%f\n", s.data[i]);

  return 0;
}

// CHECK: struct variable stride (from):
// CHECK-NEXT: 0.000000
// CHECK-NEXT: 11.000000
// CHECK-NEXT: 2.000000
// CHECK-NEXT: 33.000000
// CHECK-NEXT: 4.000000
// CHECK-NEXT: 55.000000
// CHECK-NEXT: 6.000000
// CHECK-NEXT: 77.000000
// CHECK-NEXT: 8.000000
// CHECK-NEXT: 9.000000
// CHECK: struct variable stride (to):
// CHECK-NEXT: 100.000000
// CHECK-NEXT: 2.000000
// CHECK-NEXT: 101.000000
// CHECK-NEXT: 6.000000
// CHECK-NEXT: 102.000000
// CHECK-NEXT: 10.000000
// CHECK-NEXT: 103.000000
// CHECK-NEXT: 14.000000
// CHECK-NEXT: 104.000000
// CHECK-NEXT: 18.000000
