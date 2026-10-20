// RUN: %libomptarget-compile-generic
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic

// The FROM of s comes from a mapper and is skipped (ref count 2 -> 1). The
// outer delete then releases s, and the deferred FROM transfer of s
// overwrites host s.p with the device address. Host s.p must be restored.
//
//   [0] delete: s         [1] mapper(m), from: s       ; processed [1], [0]
//   [1]: ref 2 -> 1, FROM of s skipped                                   (1)
//   [0]: ref -> 0, deferred FROM of s; host s.p = device address         (2)
//   after synchronization: host s.p = &a[0]                              (3)

#include <stdio.h>

typedef struct {
  int *p;
} S;
#pragma omp declare mapper(m : S s) map(tofrom : s, s.p[0 : 1])

int a[1];

int main() {
  S s = {a};
#pragma omp target enter data map(mapper(m), to : s)
#pragma omp target enter data map(mapper(m), to : s)

#pragma omp target exit data map(delete : s) map(mapper(m), from : s)

  // CHECK: s.p restored: 1
  printf("s.p restored: %d\n", s.p == a);
#pragma omp target exit data map(delete : a)
  return 0;
}
