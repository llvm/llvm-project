// RUN: %libomptarget-compile-generic -fopenmp-version=61
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic

// Two FROM transfers of one construct overlap the attached pointer s.p. Each
// host pointer is saved before the first transfer that covers it: a save
// before a later transfer could see the device address that an earlier one
// already wrote.
//
//   map(always, from : s, s.p)        ; processed s.p, then s
//   FROM s.p: save host s.p = &a[1]                                    (1)
//   FROM s:   s.p already saved by (1); save host s.q = &b[1]          (2)
//   after synchronization: s.p = &a[1], s.q = &b[1]                    (3)

#include <stdio.h>

typedef struct {
  int *p;
  int x;
  int *q;
} S;

int a[2], b[2];

int main() {
  S s = {&a[0], 0, &b[0]};
#pragma omp target enter data map(to : a, b, s, s.p[0 : 1], s.q[0 : 1])
#pragma omp target map(alloc : s)
  {
    s.x = 1;
  }
  s.p = &a[1];
  s.q = &b[1];
#pragma omp target map(always, from : s, s.p)
  {
  }
  // CHECK: s.p kept: 1, s.q kept: 1, s.x: 1
  printf("s.p kept: %d, s.q kept: %d, s.x: %d\n", s.p == &a[1], s.q == &b[1],
         s.x);

  s.p = &a[0];
  s.q = &b[0];
#pragma omp target exit data map(delete : s, a, b)
  return 0;
}
