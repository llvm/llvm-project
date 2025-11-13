// RUN: %libomptarget-compile-generic -fopenmp-version=60
// RUN: env LIBOMPTARGET_DEBUG=1 %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefix=DEBUG -check-prefix=CHECK
// REQUIRES: libomptarget-debug

#include <omp.h>
#include <stdio.h>

// This test shows that when an "release" is being handled, we need to look at
// all previously skipped data-transfers within the range of the memory being
// released. Here, when "s[:]" is being released, we need to honor the two
// "from" on sp1->x and sp1->y.

typedef struct {
  int x;
  int y;
} S;

int main() {
  S s[10];
  s[1].x = 111;
  s[1].y = 111;

  S *sp1 = &s[1];

#pragma omp target map(alloc : s[ : ]) map(from : sp1 -> x, sp1->y)            \
    map(to : sp1->x, sp1->y) map(alloc : s[ : ])
  {
    fprintf(stderr, "%d %d\n", s[1].x, s[1].y); // CHECK: 111 111
    s[1].x = s[1].x + 111;
    s[1].y = s[1].y + 111;
  }

  // DEBUG:      omptarget --> Found skipped FROM entry:
  // DEBUG-SAME:               HstPtr=0x[[#%x,HOST_ADDRX:]] size=[[#%u,SIZE:]]
  // DEBUG-SAME:               within region being deleted
  // DEBUG:      omptarget --> Moving [[#SIZE]] bytes (tgt:0x{{.*}}) ->
  // DEBUG-SAME:               (hst:0x{{0*}}[[#HOST_ADDRX]])
  // DEBUG:      omptarget --> Found skipped FROM entry:
  // DEBUG-SAME:               HstPtr=0x[[#%x,HOST_ADDRY:]] size=[[#%u,SIZE:]]
  // DEBUG-SAME:               within region being deleted
  // DEBUG:      omptarget --> Moving [[#SIZE]] bytes (tgt:0x{{.*}}) ->
  // DEBUG-SAME:               (hst:0x{{0*}}[[#HOST_ADDRY]])
  fprintf(stderr, "%d %d\n", s[1].x, s[1].y); // CHECK: 222 222
}
