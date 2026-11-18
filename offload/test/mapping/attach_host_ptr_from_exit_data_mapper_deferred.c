// RUN: %libomptarget-compile-generic -fopenmp-version=61
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic

// The FROM of s comes from a mapper and is skipped (ref count 2 -> 1). The
// DELETE through the mapper then releases s, and the deferred FROM transfer of
// s overwrites host s.p with the device address. Host s.p must keep its
// re-pointed value (allowed since OpenMP 6.1), while s.x is copied back.
//
//   device s.x = 1; host s.p = &a[1]
//   exit data map(mapper(m), delete : s) map(mapper(m), from : s)
//   [1] from:   ref 2 -> 1, FROM of s skipped                          (1)
//   [0] delete: ref -> 0, deferred FROM of s; host s.p = device addr   (2)
//   after synchronization: host s.p = &a[1], s.x = 1                   (3)

#include <stdio.h>

typedef struct {
  int *p;
  int x;
} S;
#pragma omp declare mapper(m : S s) map(tofrom : s, s.p[0 : 1])

int a[2];

int main() {
  S s = {&a[0], 0};
#pragma omp target enter data map(to : a)
#pragma omp target enter data map(mapper(m), to : s)
#pragma omp target enter data map(mapper(m), to : s)
#pragma omp target map(alloc : s)
  {
    s.x = 1;
  }
  s.p = &a[1];

#pragma omp target exit data map(mapper(m), delete : s) map(mapper(m), from : s)

  // CHECK: s.p kept: 1, s.x: 1
  printf("s.p kept: %d, s.x: %d\n", s.p == &a[1], s.x);
#pragma omp target exit data map(delete : a)
  return 0;
}
