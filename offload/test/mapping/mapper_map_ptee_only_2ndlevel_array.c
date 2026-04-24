// RUN: %libomptarget-compile-run-and-check-generic

#include <omp.h>
#include <stdio.h>

// Test that a mapper on a nested struct maps the right members when applied to
// an array of structs: s.x and s.p[0:10] are mapped; s.dummy and s.p itself
// are not (modulo attach FIXME).

int x[2][10];

typedef struct {
  int x;
  int dummy[10000];
  int *p;
} S1;

typedef struct {
  S1 s1;
} S2;

#pragma omp declare mapper(default : S2 s2) map(s2.s1.x, s2.s1.p[0 : 10])

S2 s2arr[2];

void print_status(void *p, const char *name) {
  int present = omp_target_is_present(p, omp_get_default_device());
  printf("%s is %spresent\n", name, present ? "" : "not ");
}

int main() {
  s2arr[0].s1.x = 111;
  s2arr[1].s1.x = 222;
  s2arr[0].s1.p = (int *)&x[0];
  s2arr[1].s1.p = (int *)&x[1];

#pragma omp target enter data map(to : s2arr)
  printf("After mapping\n");
  print_status(&s2arr[0].s1.x, "s2arr[0].s1.x"); // CHECK: s2arr[0].s1.x is present
  // dummy/p being present is not ideal, but that's what we get with the
  // current implementation because we need to map the full contiguous
  // chunk for the array first before invoking the mapper.
  print_status(&s2arr[0].s1.dummy,
               "s2arr[0].s1.dummy"); // CHECK: s2arr[0].s1.dummy is present
  print_status(&s2arr[0].s1.p,
               "s2arr[0].s1.p"); // CHECK: s2arr[0].s1.p is present
  print_status(&s2arr[0].s1.p[0],
               "s2arr[0].s1.p[0]"); // CHECK: s2arr[0].s1.p[0] is present
  print_status(&s2arr[1].s1.x, "s2arr[1].s1.x"); // CHECK: s2arr[1].s1.x is present
  print_status(&s2arr[1].s1.dummy,
               "s2arr[1].s1.dummy"); // CHECK: s2arr[1].s1.dummy is present
  print_status(&s2arr[1].s1.p,
               "s2arr[1].s1.p"); // CHECK: s2arr[1].s1.p is present
  print_status(&s2arr[1].s1.p[0],
               "s2arr[1].s1.p[0]"); // CHECK: s2arr[1].s1.p[0] is present

  printf("\n");
#pragma omp target exit data map(delete : s2arr)
  printf("After deleting\n");
  print_status(&s2arr[0].s1.x,
               "s2arr[0].s1.x"); // CHECK: s2arr[0].s1.x is not present
  print_status(&s2arr[0].s1.dummy,
               "s2arr[0].s1.dummy"); // CHECK: s2arr[0].s1.dummy is not present
  print_status(&s2arr[0].s1.p,
               "s2arr[0].s1.p"); // CHECK: s2arr[0].s1.p is not present
  print_status(&s2arr[0].s1.p[0],
               "s2arr[0].s1.p[0]"); // CHECK: s2arr[0].s1.p[0] is not present
  print_status(&s2arr[1].s1.x,
               "s2arr[1].s1.x"); // CHECK: s2arr[1].s1.x is not present
  print_status(&s2arr[1].s1.dummy,
               "s2arr[1].s1.dummy"); // CHECK: s2arr[1].s1.dummy is not present
  print_status(&s2arr[1].s1.p,
               "s2arr[1].s1.p"); // CHECK: s2arr[1].s1.p is not present
  print_status(&s2arr[1].s1.p[0],
               "s2arr[1].s1.p[0]"); // CHECK: s2arr[1].s1.p[0] is not present
}
