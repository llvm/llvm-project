// RUN: %libomptarget-compile-generic
// RUN: %libomptarget-run-fail-generic 2>&1 | %fcheck-generic

#include <omp.h>
#include <stdio.h>

// Check that the present check fails if we map a struct member, then do a
// map(present) on a mapper that maps both a member and a pointee.

int x[10];

typedef struct {
  int x;
  int dummy[10000];
  int *p;
} S;

#pragma omp declare mapper(default : S s) map(s.x, s.p[0 : 10])

S s1;

void print_status(void *p, const char *name) {
  int present = omp_target_is_present(p, omp_get_default_device());
  fprintf(stderr, "%s is %spresent\n", name, present ? "" : "not ");
}

int main() {
  s1.p = (int *)&x;

  // CHECK: addr=0x[[#%x,HOST_ADDR:]], size=[[#%u,SIZE:]]
  fprintf(stderr, "addr=%p, size=%ld\n", &s1.p[0], 10 * sizeof(s1.p[0]));

#pragma omp target enter data map(alloc : s1.x)
  printf("After mapping\n");
  print_status(&s1.x, "x");         // CHECK: x is present
  print_status(&s1.dummy, "dummy"); // CHECK: dummy is not present
  print_status(&s1.p, "p");         // CHECK: p is not present
  print_status(&s1.p[0], "p[0]");   // CHECK: p[0] is not present
  printf("\n");

  // This present check should fail!

  // clang-format off
  // CHECK: omptarget message: device mapping required by 'present' map type modifier does not exist for host address 0x{{0*}}[[#HOST_ADDR]] ([[#SIZE]] bytes)
  // CHECK: omptarget fatal error 1: failure of target construct while offloading is mandatory
  // clang-format on

#pragma omp target enter data map(present, alloc : s1)
}
