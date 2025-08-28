// RUN: %libomptarget-compilexx-run-and-check-generic

// REQUIRES: libc

#include <omp.h>
#include <stdio.h>

typedef struct {
  short x;
  int *p;
  long y;
} S;

S s[10], *ps;

void f1() {
  ps = &s[0];
  s[0].x = 111;
  s[1].x = 222;
  s[2].x = 333;
  s[3].x = 444;

#pragma omp target enter data map(to : s)
#pragma omp target enter data map(to : ps, ps->x)

  S **ps_mappedptr = (S **)omp_get_mapped_ptr(&ps, omp_get_default_device());
  short *s0_mappedptr =
      (short *)omp_get_mapped_ptr(&s[0].x, omp_get_default_device());
  short *s0_hostaddr = &s[0].x;

  printf("ps_mappedptr %s null\n", ps_mappedptr == (S **)NULL ? "==" : "!=");
  printf("s0_mappedptr %s null\n", s0_mappedptr == (short *)NULL ? "==" : "!=");

// CHECK: ps_mappedptr != null
// CHECK: s0_mappedptr != null

// ps is predetermined firstprivate, so its address will be different from
// the mapped address for this construct. So, any changes to p within the
// region will not be visible after the construct.
#pragma omp target map(ps->x) map(to : ps_mappedptr, s0_mappedptr, s0_hostaddr)
  {
    printf("%d %d %d %d\n", ps->x, ps_mappedptr == &ps, s0_mappedptr == &ps->x,
           s0_hostaddr == &ps->x);
    // CHECK: 111 0 1 0
    ps++;
  }

// For the remaining constructs, ps is not firstprivate, so its address will
// be the same as the mapped address, and changes to ps will be visible to any
// subsequent regions.
#pragma omp target map(to : ps->x, ps)                                         \
    map(to : ps_mappedptr, s0_mappedptr, s0_hostaddr)
  {
    printf("%d %d %d %d\n", ps->x, ps_mappedptr == &ps, s0_mappedptr == &ps->x,
           s0_hostaddr == &ps->x);
    // EXPECTED: 111 1 1 0
    // CHECK:    111 0 1 0
    ps++;
  }

#pragma omp target map(to : ps, ps->x)                                         \
    map(to : ps_mappedptr, s0_mappedptr, s0_hostaddr)
  {
    printf("%d %d %d %d\n", ps->x, ps_mappedptr == &ps,
           s0_mappedptr == &ps[-1].x, s0_hostaddr == &ps[-1].x);
    // EXPECTED: 222 1 1 0
    // CHECK:    111 0 0 0
    ps++;
  }

#pragma omp target map(present, alloc : ps)                                    \
    map(to : ps_mappedptr, s0_mappedptr, s0_hostaddr)
  {
    printf("%d %d %d %d\n", ps->x, ps_mappedptr == &ps,
           s0_mappedptr == &ps[-2].x, s0_hostaddr == &ps[-2].x);
    // EXPECTED: 333 1 1 0
    // CHECK:    111 1 0 0
  }

  // The following map(from:ps) should not bring back ps, because ps is an
  // attached pointer. So, it should still point to the same original
  // location, &s[0], on host.
#pragma omp target exit data map(always, from : ps)
  printf("%d %d\n", ps->x, ps == &s[0]);
  // CHECK: 111 1

#pragma omp target exit data map(delete : ps, s)
}

int main() { f1(); }
