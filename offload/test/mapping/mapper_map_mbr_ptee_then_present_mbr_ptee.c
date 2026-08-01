// Check that it's ok to first map a member of a struct and its pointee, and
// then do a map(present) on a mapper that maps them internally.
//
// FIXME: This currently run-fails, because the mapper does not yet emit
// attach-style maps for the pointee: the combined entry over the whole struct
// s1 (40016 bytes) triggers an "explicit extension not allowed" error against
// the 4-byte device allocation of s1.x. Once attach-style maps are emitted for
// the pointee, the present check should pass and the run should complete the
// present/delete sequence below.
// RUN: %libomptarget-compile-generic
// RUN: %libomptarget-run-fail-generic 2>&1 \
// RUN: | %fcheck-generic --check-prefix=CHECK

#include <omp.h>
#include <stdio.h>

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
  // EXPECTED: After mapping
  print_status(&s1.x, "x");         // EXPECTED: x is present
  print_status(&s1.dummy, "dummy"); // EXPECTED: dummy is not present
  print_status(&s1.p, "p");         // EXPECTED: p is not present
  print_status(&s1.p[0], "p[0]");   // EXPECTED: p[0] is present

  // This present check currently fails (explicit extension); once attach-style
  // maps are emitted for the pointee, it should pass.
  // clang-format off
  // CHECK: omptarget message: explicit extension not allowed
  // CHECK: omptarget fatal error 1: failure of target construct while offloading is mandatory
  // clang-format on
#pragma omp target enter data map(present, alloc : s1)

#pragma omp target exit data map(delete : s1)
  // EXPECTED: After deleting
  print_status(&s1.x, "x");         // EXPECTED: x is not present
  print_status(&s1.dummy, "dummy"); // EXPECTED: dummy is not present
  print_status(&s1.p, "p");         // EXPECTED: p is not present
  print_status(&s1.p[0], "p[0]");   // EXPECTED: p[0] is not present
}
