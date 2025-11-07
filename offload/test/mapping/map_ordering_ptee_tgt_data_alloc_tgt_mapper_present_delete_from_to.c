// RUN: %libomptarget-compile-generic -fopenmp-version=60
// RUN: %libomptarget-run-generic | %fcheck-generic

// The "present" check should pass on the "target" construct // (2),
// and there should be no "to" transfer, because the pointee "x" is already
// present (because of (1)).
// However, there should be a "from" transfer at the end of (2) because of the
// "delete" on the mapper.

// This currently fails, but should start passing once ATTACH-style maps are
// enabled for mappers (#166874).
// XFAIL: *

#include <stdio.h>

typedef struct {
  int *p;
  int *q;
} S;
#pragma omp declare mapper(my_mapper: S s) map(alloc: s.p) map(alloc, present: s.p[0:10]) map(delete: s.q[:]) map(from: s.p[0:10]) map(to: s.p[0:10]) map(alloc: s.p[0:10])

S s1;
int main() {
  int x[10];
  x[1] = 111;
  s1.q = s1.p = &x[0];

  #pragma omp target data map(alloc: x) // (1)
  {
    #pragma omp target map(mapper(my_mapper), tofrom: s1) // (2)
    {
      printf("%d\n", s1.p[1]); // CHECK-NOT: 111
      s1.p[1] = 222;
    }
    printf("%d\n", s1.p[1]); // CHECK: 222
  }
}
