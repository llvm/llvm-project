// RUN: %libomptarget-compile-generic -fopenmp-version=60
// RUN: %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefix=CHECK
// RUN: env LIBOMPTARGET_DEBUG=1 %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefix=DEBUG
// REQUIRES: libomptarget-debug

#include <omp.h>
#include <stdio.h>

// This test shows that when a "release" is being handled, we look at
// all previously skipped data-transfers within the range of the memory being
// released. Here, when "s[:]" is being released, we honor the two "from"s
// on sp1->x and sp1->y.

typedef struct {
  int x;
  int y;
} S;

int main() {
  S s[10];
  s[1].x = 111;
  s[1].y = 111;

  S *sp1 = &s[1];

// clang-format off
// DEBUG: omptarget --> HstPtrBegin 0x[[#%x,HOST_ADDRX:]] was newly allocated for the current region
// DEBUG: omptarget --> Moving [[#%u,SIZEX:]] bytes (hst:0x{{0*}}[[#HOST_ADDRX]]) -> (tgt:0x{{.*}})
// DEBUG: omptarget --> HstPtrBegin 0x[[#%x,HOST_ADDRY:]] was newly allocated for the current region
// DEBUG: omptarget --> Moving [[#%u,SIZEY:]] bytes (hst:0x{{0*}}[[#HOST_ADDRY]]) -> (tgt:0x{{.*}})
// clang-format on
#pragma omp target map(alloc : s[ : ]) map(from : sp1 -> x, sp1->y)            \
    map(to : sp1->x, sp1->y) map(alloc : s[ : ])
  {
    fprintf(stderr, "%d %d\n", s[1].x, s[1].y); // CHECK: 111 111
    s[1].x = s[1].x + 111;
    s[1].y = s[1].y + 111;
  }

  // clang-format off
  // DEBUG: omptarget --> Found skipped FROM entry: HstPtr=0x{{0*}}[[#HOST_ADDRY]] size=[[#SIZEY]] within region being deleted
  // DEBUG: omptarget --> Moving [[#SIZEY]] bytes (tgt:0x{{.*}}) -> (hst:0x{{0*}}[[#HOST_ADDRY]])
  // DEBUG: omptarget --> Found skipped FROM entry: HstPtr=0x{{0*}}[[#HOST_ADDRX]] size=[[#SIZEX]] within region being deleted
  // DEBUG: omptarget --> Moving [[#SIZEX]] bytes (tgt:0x{{.*}}) -> (hst:0x{{0*}}[[#HOST_ADDRX]])
  // clang-format on
  fprintf(stderr, "%d %d\n", s[1].x, s[1].y); // CHECK: 222 222
}
