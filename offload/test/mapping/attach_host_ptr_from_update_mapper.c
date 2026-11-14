// RUN: %libomptarget-compile-generic -fopenmp-version=61
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic

// A target update from with a mapper transfers the mapper's components through
// a recursive targetDataUpdate. The FROM of s overwrites host s.p with the
// device address, and host s.p must keep its value from before the transfer.

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
#pragma omp target map(alloc : s)
  {
    s.x = 1;
  }
  s.p = &a[1];
#pragma omp target update from(mapper(m) : s)
  // CHECK: s.p kept: 1, s.x: 1
  printf("s.p kept: %d, s.x: %d\n", s.p == &a[1], s.x);

  s.p = &a[0];
#pragma omp target exit data map(mapper(m), delete : s)
#pragma omp target exit data map(delete : a)
  return 0;
}
