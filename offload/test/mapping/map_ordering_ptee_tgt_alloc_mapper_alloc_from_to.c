// RUN: %libomptarget-compile-generic
// RUN: %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefix=CHECK
// RUN: env LIBOMPTARGET_DEBUG=1 %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefix=DEBUG
// REQUIRES: libomptarget-debug

// Since the allocation of the pointee happens on the "target" construct (1),
// the "to" transfer requested as part of the mapper (2) should also happen.
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
    map(from : s.p[0 : 10]) map(to : s.p[0 : 10])                              \
    map(alloc : s.p[0 : 10]) // (2)

S s1;
int main() {
  int x[10];
  x[1] = 111;
  s1.q = s1.p = &x[0];

  // clang-format off
  // DEBUG: omptarget --> HstPtrBegin 0x[[#%x,HOST_ADDRX:]] was newly allocated for the current region
  // DEBUG: omptarget --> Moving [[#%u,SIZEX:]] bytes (hst:0x{{0*}}[[#HOST_ADDRX]]) -> (tgt:0x{{.*}})
  // clang-format on
#pragma omp target map(alloc : s1.p[0 : 10])                                   \
    map(mapper(my_mapper), tofrom : s1) // (1)
  {
    printf("%d\n", s1.p[1]); // CHECK: 111
    s1.p[1] = s1.p[1] + 111;
  }

  // clang-format off
  // DEBUG: omptarget --> Found skipped FROM entry: HstPtr=0x{{0*}}[[#HOST_ADDRX]] size=[[#SIZEX]] within region being released
  // DEBUG: omptarget --> Moving [[#SIZEX]] bytes (tgt:0x{{.*}}) -> (hst:0x{{0*}}[[#HOST_ADDRX]])
  // clang-format on
  printf("%d\n", s1.p[1]); // CHECK: 222
}
