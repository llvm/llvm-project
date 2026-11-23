// RUN: %libomptarget-compile-run-and-check-generic

#include <omp.h>
#include <stdio.h>

// Array variant of mapper_map_ptee_only_2_ptr_indirections.c.
// The mapper maps s2.z, s2.s1p->x, s2.s1p->y, and s2.s1p->p[0:10].
// s2.s1p->dummy and s2.s1p->p itself are not mapped.
// This exercises the nested-pointer-chain case: the inner MEMBER_OF bits for
// s2.s1p->x/y/p[0:10] must be shifted correctly, and outer MEMBER_OF must not
// be applied to the pointee entry (s2.s1p->p[0:10]) or the ATTACH entry
// (s2.s1p).

int x[2][10];

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

S1 s1arr[2];
S2 s2arr[2];

void print_status(void *p, const char *name) {
  int present = omp_target_is_present(p, omp_get_default_device());
  printf("%s is %spresent\n", name, present ? "" : "not ");
}

int main() {
  s2arr[0].s1p = &s1arr[0];
  s2arr[1].s1p = &s1arr[1];
  s2arr[0].s1p->p = (int *)&x[0];
  s2arr[1].s1p->p = (int *)&x[1];

#pragma omp target enter data map(alloc : s2arr)
  printf("After mapping\n");
  print_status(&s2arr[0].s1p->x, "s2arr[0].x"); // CHECK: s2arr[0].x is present
  print_status(&s2arr[0].s1p->y, "s2arr[0].y"); // CHECK: s2arr[0].y is present
  print_status(&s2arr[0].z, "s2arr[0].z");      // CHECK: s2arr[0].z is present
  print_status(&s2arr[0].s1p->dummy,
               "s2arr[0].dummy"); // CHECK: s2arr[0].dummy is not present
  print_status(&s2arr[0].s1p->p,
               "s2arr[0].p"); // CHECK: s2arr[0].p is not present
  print_status(&s2arr[0].s1p->p[0],
               "s2arr[0].p[0]"); // CHECK: s2arr[0].p[0] is present
  print_status(&s2arr[1].s1p->x, "s2arr[1].x"); // CHECK: s2arr[1].x is present
  print_status(&s2arr[1].s1p->y, "s2arr[1].y"); // CHECK: s2arr[1].y is present
  print_status(&s2arr[1].z, "s2arr[1].z");      // CHECK: s2arr[1].z is present
  print_status(&s2arr[1].s1p->dummy,
               "s2arr[1].dummy"); // CHECK: s2arr[1].dummy is not present
  print_status(&s2arr[1].s1p->p,
               "s2arr[1].p"); // CHECK: s2arr[1].p is not present
  print_status(&s2arr[1].s1p->p[0],
               "s2arr[1].p[0]"); // CHECK: s2arr[1].p[0] is present
  printf("\n");

#pragma omp target exit data map(delete : s2arr)
  printf("After deleting\n");
  print_status(&s2arr[0].s1p->x,
               "s2arr[0].x"); // CHECK: s2arr[0].x is not present
  print_status(&s2arr[0].s1p->y,
               "s2arr[0].y");              // CHECK: s2arr[0].y is not present
  print_status(&s2arr[0].z, "s2arr[0].z"); // CHECK: s2arr[0].z is not present
  print_status(&s2arr[0].s1p->dummy,
               "s2arr[0].dummy"); // CHECK: s2arr[0].dummy is not present
  print_status(&s2arr[0].s1p->p,
               "s2arr[0].p"); // CHECK: s2arr[0].p is not present
  print_status(&s2arr[0].s1p->p[0],
               "s2arr[0].p[0]"); // CHECK: s2arr[0].p[0] is not present
  print_status(&s2arr[1].s1p->x,
               "s2arr[1].x"); // CHECK: s2arr[1].x is not present
  print_status(&s2arr[1].s1p->y,
               "s2arr[1].y");              // CHECK: s2arr[1].y is not present
  print_status(&s2arr[1].z, "s2arr[1].z"); // CHECK: s2arr[1].z is not present
  print_status(&s2arr[1].s1p->dummy,
               "s2arr[1].dummy"); // CHECK: s2arr[1].dummy is not present
  print_status(&s2arr[1].s1p->p,
               "s2arr[1].p"); // CHECK: s2arr[1].p is not present
  print_status(&s2arr[1].s1p->p[0],
               "s2arr[1].p[0]"); // CHECK: s2arr[1].p[0] is not present
}
