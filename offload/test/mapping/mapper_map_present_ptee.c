// A user-defined mapper maps s.y and, with present written directly in the
// mapper's own clause, the pointee s.p. present in the mapper clause applies at
// every OpenMP version -- this is not the outer-clause propagation that a
// follow-on gates on the version, so both 5.2 and 6.0 behave the same here.
//
// RUN: %libomptarget-compile-run-and-check-generic
// RUN: %libomptarget-compile-generic -DOUT_OF_BOUNDS
// RUN: %libomptarget-run-fail-generic 2>&1 \
// RUN: | %fcheck-generic --check-prefix=CHECK-OOB

#include <stdio.h>

int x[10];
typedef struct {
  int y;
  int *p;
} S;

#ifdef OUT_OF_BOUNDS
// s.p[0:20] extends beyond the mapped region x[0:10]; the present check fails.
#pragma omp declare mapper(S s) map(s.y) map(present, tofrom : s.p[0 : 20])
#else
#pragma omp declare mapper(S s) map(s.y) map(present, tofrom : s.p[0 : 2])
#endif
S s;

void f1() {
  // The mapper runs here; for OUT_OF_BOUNDS the present check fails here.
#pragma omp target update to(s)

#pragma omp target data use_device_addr(s, x)
#pragma omp target has_device_addr(s, x)
  {
    s.y = s.y + 222;
    x[0] = x[0] + 222;
  }
}

int main() {
  x[0] = 111;
  s.y = 111;
  s.p = &x[0];

  // CHECK-OOB: addr=0x[[#%x,HOST_ADDR:]], size=[[#%u,SIZE:]]
  fprintf(stderr, "addr=%p, size=%zu\n", &s.p[0], 20 * sizeof(s.p[0]));

#pragma omp target data map(from : s.y, x)
  {
    f1();
  }

  printf("%d %d\n", x[0], s.y); // CHECK: 333 333
  // clang-format off
  // CHECK-OOB: message: device mapping required by 'present' motion modifier does not exist for host address 0x{{0*}}[[#HOST_ADDR]] ([[#SIZE]] bytes)
  // CHECK-OOB: fatal error 1: failure of target construct while offloading is mandatory
  // clang-format on
}
