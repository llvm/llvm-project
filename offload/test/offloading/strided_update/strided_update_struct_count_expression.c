// RUN: %libomptarget-compile-run-and-check-generic
// Tests non-contiguous array sections with expression-based count on struct
// member arrays with both FROM and TO directives.

#include <omp.h>
#include <stdio.h>

struct S {
  int len;
  double data[20];
};

int main() {
  struct S s;
  s.len = 10;

  // Initialize on device
#pragma omp target map(tofrom : s)
  {
    for (int i = 0; i < s.len; i++) {
      s.data[i] = i;
    }
  }

  // Test FROM: Modify on device, then update from device
#pragma omp target data map(to : s)
  {
#pragma omp target
    {
      for (int i = 0; i < s.len; i++) {
        s.data[i] += i * 10;
      }
    }

    // Update from device with expression-based count: len/2 elements
#pragma omp target update from(s.data[0 : s.len / 2 : 2])
  }

  printf("struct count expression (from):\n");
  for (int i = 0; i < s.len; i++)
    printf("%f\n", s.data[i]);

  // Test TO: Reset, modify host, update to device
#pragma omp target map(tofrom : s)
  {
    for (int i = 0; i < s.len; i++) {
      s.data[i] = i * 2;
    }
  }

  // Modify host data
  for (int i = 0; i < s.len / 2; i++) {
    s.data[i * 2] = i + 100;
  }

  // Update to device with expression-based count
#pragma omp target data map(alloc : s)
  {
#pragma omp target update to(s.data[0 : s.len / 2 : 2])

#pragma omp target
    {
      for (int i = 0; i < s.len; i++) {
        s.data[i] += 100;
      }
    }
  }

  printf("struct count expression (to):\n");
  for (int i = 0; i < s.len; i++)
    printf("%f\n", s.data[i]);

  return 0;
}

// CHECK: struct count expression (from):
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
// CHECK: struct count expression (to):
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
