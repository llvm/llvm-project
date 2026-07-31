// The mapper maps a struct member (s.x) and a pointee (s.p[0:10]). We pre-map
// only s.x, then do map(present) on the mapper. The pointee s.p[0:10] is not
// present, so once PRESENT is propagated to the pointee (a follow-on, at OpenMP
// >= 6.0) the check must fail; at <= 5.2 present is not propagated, so it would
// pass.
//
// FIXME: This currently run-fails at BOTH versions, because the mapper does not
// yet emit attach-style maps for the pointee: the combined entry over the whole
// struct s1 (40016 bytes) triggers an "explicit extension not allowed" error
// against the 4-byte device allocation of s1.x. Once attach-style maps are
// emitted for the pointee:
//   EXPECTED (5.2): the run completes ("done").
//   EXPECTED (6.0): the present check fails for the absent pointee s1.p[0:10].
// RUN: %libomptarget-compile-generic -fopenmp-version=52
// RUN: %libomptarget-run-fail-generic 2>&1 \
// RUN: | %fcheck-generic --check-prefixes=CHECK
// RUN: %libomptarget-compile-generic -fopenmp-version=60
// RUN: %libomptarget-run-fail-generic 2>&1 \
// RUN: | %fcheck-generic --check-prefixes=CHECK

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
  fprintf(stderr, "%s is %spresent\n", name, present ? "" : "not ");
}

int main() {
  s1.p = (int *)&x;

  fprintf(stderr, "addr=%p, size=%ld\n", &s1.p[0], 10 * sizeof(s1.p[0]));

#pragma omp target enter data map(alloc : s1.x)
  print_status(&s1.x, "x");         // EXPECTED: x is present
  print_status(&s1.dummy, "dummy"); // EXPECTED: dummy is not present
  print_status(&s1.p, "p");         // EXPECTED: p is not present
  print_status(&s1.p[0], "p[0]");   // EXPECTED: p[0] is not present

#pragma omp target enter data map(present, alloc : s1)
  // Once attach-style maps are emitted for the pointee, at 5.2 the run
  // completes past this point; at 6.0 the present check on the absent pointee
  // s1.p[0:10] fails here.
  // clang-format off
  // CHECK: omptarget message: explicit extension not allowed
  // CHECK: omptarget fatal error 1: failure of target construct while offloading is mandatory
  // clang-format on

  fprintf(stderr, "done\n");
}
