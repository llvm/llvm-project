// RUN: %libomptarget-compile-run-and-check-generic

// Since the allocation of the pointee happens as on the "target" construct (1),
// the "to" transfer requested as part of the mapper should also happen.
//
// Similarly, the "from" transfer should also happen at the end of the target
// construct, even if the ref-count of the pointee x has not gone down to 0
// when "from" is encountered.

#include <stdio.h>

typedef struct {
  int *p;
  int *q;
} S;
#pragma omp declare mapper(my_mapper : S s) map(alloc : s.p, s.p[0 : 10])      \
    map(from : s.p[0 : 10]) map(to : s.p[0 : 10]) map(alloc : s.p[0 : 10])

S s1;
int main() {
  int x[10];
  x[1] = 111;
  s1.q = s1.p = &x[0];

#pragma omp target map(alloc : s1.p[0 : 10])                                   \
    map(mapper(my_mapper), tofrom : s1) // (1)
  {
    printf("%d\n", s1.p[1]); // CHECK: 111
    s1.p[1] = s1.p[1] + 111;
  }

  printf("%d\n", s1.p[1]); // CHECK: 222
}
