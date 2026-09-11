// RUN: %libomptarget-compile-run-and-check-generic

#include <omp.h>
#include <stdio.h>

// Test that a mapper that maps a var and a pointee, correctly propagates the
// always and present bits into the individual maps "pushed" by it.

int x[2][10];

typedef struct {
  int x;
  int dummy[10000];
  int *p;
} S;

#pragma omp declare mapper(default : S s) map(s.x, s.p[0 : 10])

S s1[2];

void print_status(void *p, const char *name) {
  int present = omp_target_is_present(p, omp_get_default_device());
  printf("%s is %spresent\n", name, present ? "" : "not ");
}

int main() {
  x[0][0] = x[0][1] = s1[0].x = s1[1].x = 111;
  s1[0].p = (int *)&x[0];
  s1[1].p = (int *)&x[1];

#pragma omp target enter data map(alloc : s1)
  printf("After mapping\n");
  print_status(&s1[0].x, "s1[0].x"); // CHECK: s1[0].x is present
  // dummy/p being present is not ideal, but that's what we get with the
  // current implementation because we need to map the full contiguous
  // chunk for the array first before invoking the mapper.
  print_status(&s1[0].dummy, "s1[0].dummy"); // CHECK: s1[0].dummy is present
  print_status(&s1[0].p, "s1[0].p");         // CHECK: s1[0].p is present
  print_status(&s1[0].p[0], "s1[0].p[0]");   // CHECK: s1[0].p[0] is present
  print_status(&s1[1].x, "s1[1].x");         // CHECK: s1[1].x is present
  print_status(&s1[1].dummy, "s1[1].dummy"); // CHECK: s1[1].dummy is present
  print_status(&s1[1].p, "s1[1].p");         // CHECK: s1[1].p is present
  print_status(&s1[1].p[0], "s1[1].p[0]");   // CHECK: s1[1].p[0] is present

#pragma omp target map(always, present, from : s1)
  {
    s1[0].p[0] = s1[1].p[0] = s1[0].x = s1[1].x = 222;
  }

  printf("\n");
  printf("After map(always,from)\n");
  printf("s[0].x = %d\n", s1[0].x);       // CHECK: s[0].x = 222
  printf("s[1].x = %d\n", s1[1].x);       // CHECK: s[1].x = 222
  printf("s[0].p[0] = %d\n", s1[0].p[0]); // CHECK: s[0].p[0] = 222
  printf("s[1].p[0] = %d\n", s1[1].p[0]); // CHECK: s[1].p[0] = 222
  printf("\n");

#pragma omp target exit data map(delete : s1)
  printf("After deleting\n");
  print_status(&s1[0].x, "s1[0].x"); // CHECK: s1[0].x is not present
  print_status(&s1[0].dummy,
               "s1[0].dummy");             // CHECK: s1[0].dummy is not present
  print_status(&s1[0].p, "s1[0].p");       // CHECK: s1[0].p is not present
  print_status(&s1[0].p[0], "s1[0].p[0]"); // CHECK: s1[0].p[0] is not present
  print_status(&s1[1].x, "s1[1].x");       // CHECK: s1[1].x is not present
  print_status(&s1[1].dummy,
               "s1[1].dummy");             // CHECK: s1[1].dummy is not present
  print_status(&s1[1].p, "s1[1].p");       // CHECK: s1[1].p is not present
  print_status(&s1[1].p[0], "s1[1].p[0]"); // CHECK: s1[1].p[0] is not present
}
