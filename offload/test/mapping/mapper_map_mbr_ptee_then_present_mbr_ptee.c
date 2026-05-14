// RUN: %libomptarget-compile-run-and-check-generic

#include <omp.h>
#include <stdio.h>

// Check that it's ok to first map member of a struct and its pointee, and then
// do a map(present) on a mapper that maps them internally.

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

#pragma omp target enter data map(alloc : s1.x, s1.p[0 : 10])
  printf("After mapping\n");
  print_status(&s1.x, "x");         // CHECK: x is present
  print_status(&s1.dummy, "dummy"); // CHECK: dummy is not present
  print_status(&s1.p, "p");         // CHECK: p is not present
  print_status(&s1.p[0], "p[0]");   // CHECK: p[0] is present
  printf("\n");

  // This present check should pass.
#pragma omp target enter data map(present, alloc : s1)

#pragma omp target exit data map(delete : s1)
  printf("After deleting\n");
  print_status(&s1.x, "x");         // CHECK: x is not present
  print_status(&s1.dummy, "dummy"); // CHECK: dummy is not present
  print_status(&s1.p, "p");         // CHECK: p is not present
  print_status(&s1.p[0], "p[0]");   // CHECK: p[0] is not present
}
