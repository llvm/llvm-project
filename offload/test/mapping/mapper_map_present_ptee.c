// A user-defined mapper maps s.y and, with present written directly in the
// mapper's own clause, the pointee s.p. present in the mapper clause applies at
// every OpenMP version -- this is not the outer-clause propagation that a
// follow-on gates on the version.
//
// FIXME: This currently run-fails at BOTH the inbounds and out-of-bounds cases,
// because the mapper does not yet emit attach-style maps for the pointee: the
// present check runs against the whole struct s (16 bytes), which is not
// present. Once attach-style maps are emitted for the pointee:
//   EXPECTED (inbounds): the present check passes and the run prints "333 333".
//   EXPECTED (out-of-bounds): the present check fails only for the pointee
//     region s.p[0:20] that is not present.
// RUN: %libomptarget-compile-generic
// RUN: %libomptarget-run-fail-generic 2>&1 \
// RUN: | %fcheck-generic --check-prefix=CHECK
// RUN: %libomptarget-compile-generic -DOUT_OF_BOUNDS
// RUN: %libomptarget-run-fail-generic 2>&1 \
// RUN: | %fcheck-generic --check-prefix=CHECK

#include <stdio.h>

int x[10];
typedef struct {
  int y;
  int *p;
} S;

#ifdef OUT_OF_BOUNDS
#pragma omp declare mapper(S s) map(s.y) map(present, tofrom : s.p[0 : 20])
#else
#pragma omp declare mapper(S s) map(s.y) map(present, tofrom : s.p[0 : 2])
#endif
S s;

void f1() {
  // The mapper runs here; the present check currently fails here at both cases.
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

  fprintf(stderr, "addr=%p, size=%zu\n", &s.p[0], 20 * sizeof(s.p[0]));

#pragma omp target data map(from : s.y, x)
  {
    f1();
  }

  // EXPECTED (inbounds): 333 333
  fprintf(stderr, "%d %d\n", x[0], s.y);
  // clang-format off
  // CHECK: omptarget message: device mapping required by 'present' motion modifier does not exist for host address
  // CHECK: omptarget fatal error 1: failure of target construct while offloading is mandatory
  // clang-format on
}
