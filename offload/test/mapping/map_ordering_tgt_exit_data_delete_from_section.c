// RUN: %libomptarget-compile-run-and-check-generic
// UNSUPPORTED: apu

// The FROM of x[1:1] is skipped (ref count 2 -> 1). The DELETE of x[1:1] then
// releases x, and the deferred FROM must copy device x[1], at offset 4 from the
// begin of x's device buffer, not x[2] at offset 8.
//
//   device x = {222, 333, 444}; host x = {111, 111, 111}
//   exit data map(delete : x[1:1]) map(from : x[1:1])
//   [1] from:   ref 2 -> 1, FROM of x[1:1] skipped                     (1)
//   [0] delete: ref -> 0, deferred FROM of x[1:1]                      (2)
//   host x = {111, 333, 111}                                           (3)

#include <stdio.h>

int main() {
  int x[3] = {111, 111, 111};
#pragma omp target enter data map(to : x)
#pragma omp target enter data map(to : x)
#pragma omp target map(alloc : x)
  {
    x[0] = 222;
    x[1] = 333;
    x[2] = 444;
  }

#pragma omp target exit data map(delete : x[1 : 1]) map(from : x[1 : 1])

  // CHECK: x: 111 333 111
  printf("x: %d %d %d\n", x[0], x[1], x[2]);
  return 0;
}
