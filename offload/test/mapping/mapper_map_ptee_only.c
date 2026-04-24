// RUN: %libomptarget-compile-run-and-check-generic

#include <omp.h>
#include <stdio.h>

// This test ensures that when mapping `s.x` and `s.p[0:10] via a mapper, the
// storage of `s.p` and hence `s.dummy` is not unnecessarily allocated.

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
  printf("%s is %spresent\n", name, present ? "" : "not ");
}

int main() {
  s1.p = (int *)&x;

#pragma omp target enter data map(alloc : s1)
  printf("After mapping\n");
  print_status(&s1.x, "x");         // CHECK: x is present
  print_status(&s1.dummy, "dummy"); // CHECK: dummy is not present
  print_status(&s1.p, "p");         // CHECK: p is not present
  print_status(&s1.p[0], "p[0]");   // CHECK: p[0] is present
  printf("\n");

#pragma omp target exit data map(delete : s1)
  printf("After deleting\n");
  print_status(&s1.x, "x");         // CHECK: x is not present
  print_status(&s1.dummy, "dummy"); // CHECK: dummy is not present
  print_status(&s1.p, "p");         // CHECK: p is not present
  print_status(&s1.p[0], "p[0]");   // CHECK: p[0] is not present
}
