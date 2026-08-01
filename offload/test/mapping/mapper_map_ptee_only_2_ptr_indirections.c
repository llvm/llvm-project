// RUN: %libomptarget-compile-run-and-check-generic

#include <omp.h>
#include <stdio.h>

// The mapper maps s2.z, s2.s1p->x, s2.s1p->y, and s2.s1p->p[0:10].
// Check that s2.s1p->dummy and s2.s1p->p itself are not mapped, and that all
// mapped fields are correctly removed on delete.

int x[10];

typedef struct {
  int x;
  int y;
  int dummy[10000];
  int *p;
} S1;

typedef struct {
  S1 *s1p;
  int z;
} S2;

#pragma omp declare mapper(default : S2 s2)                                    \
    map(s2.z, s2.s1p -> x, s2.s1p->y, s2.s1p->p[0 : 10])

S1 s1;
S2 s2;

void print_status(void *p, const char *name) {
  int present = omp_target_is_present(p, omp_get_default_device());
  printf("%s is %spresent\n", name, present ? "" : "not ");
}

int main() {
  s2.s1p = &s1;
  s2.s1p->p = (int *)&x;

#pragma omp target enter data map(alloc : s2)
  printf("After mapping\n");
  print_status(&s2.s1p->x, "x");         // CHECK: x is present
  print_status(&s2.s1p->y, "y");         // CHECK: y is present
  print_status(&s2.z, "z");              // CHECK: z is present
  print_status(&s2.s1p->dummy, "dummy"); // CHECK: dummy is not present
  print_status(&s2.s1p->p, "p");         // CHECK: p is present
  // FIXME: mapper should emit attach-style maps for pointer members.
  //                                        EXPECTED: p is not present
  print_status(&s2.s1p->p[0], "p[0]"); // CHECK: p[0] is present
  printf("\n");

#pragma omp target exit data map(delete : s2)
  printf("After deleting\n");
  print_status(&s2.s1p->x, "x");         // CHECK: x is not present
  print_status(&s2.s1p->y, "y");         // CHECK: y is not present
  print_status(&s2.z, "z");              // CHECK: z is not present
  print_status(&s2.s1p->dummy, "dummy"); // CHECK: dummy is not present
  print_status(&s2.s1p->p, "p");         // CHECK: p is present
  // FIXME: the DELETE modifier is not propagated to mapper entries yet.
  //                                        EXPECTED: p is not present
  print_status(&s2.s1p->p[0], "p[0]"); // CHECK: p[0] is not present
}
