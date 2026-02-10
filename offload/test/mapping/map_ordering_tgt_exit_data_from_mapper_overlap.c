// RUN: %libomptarget-compile-generic
// RUN: %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefix=CHECK
// RUN: env LIBOMPTARGET_DEBUG=1 %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefix=DEBUG
// REQUIRES: libomptarget-debug

// The test ensures that the FROM transfer for the full "s1" is performed, and
// not just the FROM done via the mapper of s1.s2.

#include <stdio.h>

typedef struct {
  int a;
  int b;
} S2;

#pragma omp declare mapper(my_mapper : S2 s2) map(tofrom : s2.a)

typedef struct {
  S2 s2;
  int c;
  int d;
} S1;

S1 s1;

int main() {
#pragma omp target enter data map(alloc : s1)

#pragma omp target map(present, alloc : s1)
  {
    s1.s2.a = 111;
    s1.s2.b = 222;
    s1.c = 333;
    s1.d = 444;
  }

  // clang-format off
  // DEBUG: omptarget --> Tracking released entry: HstPtr=0x[[#%x,HOST_ADDR:]], Size=[[#%u,SIZE:]], ForceDelete=0
  // DEBUG: omptarget --> Moving {{.*}} bytes (tgt:0x{{.*}}) -> (hst:0x{{.*}})
  // DEBUG: omptarget --> Pointer HstPtr=0x{{0*}}[[#HOST_ADDR]] falls within a range previously released
  // DEBUG: omptarget --> Moving [[#SIZE]] bytes (tgt:0x{{.*}}) -> (hst:0x{{0*}}[[#HOST_ADDR]])
  // clang-format on
#pragma omp target exit data map(from : s1) map(mapper(my_mapper), from : s1.s2)

  // CHECK: 111 222 333 444
  printf("%d %d %d %d\n", s1.s2.a, s1.s2.b, s1.c, s1.d);
}
