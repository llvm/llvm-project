// The mapper maps a struct member (s.x) and a pointee (s.p[0:10]). We pre-map
// only s.x, then do map(present) on the mapper. The pointee s.p[0:10] is not
// present, so the propagated present modifier must fail the check -- but only
// at OpenMP >= 6.0, since present is not propagated to the pointee before then.

// OpenMP <= 5.2: present is not propagated to the pointee, so the run succeeds.
// RUN: %libomptarget-compile-generic -fopenmp-version=52
// RUN: %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic --check-prefixes=CHECK,CHECK-52

// OpenMP 6.0: present is propagated to the pointee; the check fails.
// RUN: %libomptarget-compile-generic -fopenmp-version=60
// RUN: %libomptarget-run-fail-generic 2>&1 \
// RUN: | %fcheck-generic --check-prefixes=CHECK,CHECK-60

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

  // CHECK: addr=0x[[#%x,HOST_ADDR:]], size=[[#%u,SIZE:]]
  fprintf(stderr, "addr=%p, size=%ld\n", &s1.p[0], 10 * sizeof(s1.p[0]));

#pragma omp target enter data map(alloc : s1.x)
  print_status(&s1.x, "x");         // CHECK: x is present
  print_status(&s1.dummy, "dummy"); // CHECK: dummy is not present
  print_status(&s1.p, "p");         // CHECK: p is not present
  print_status(&s1.p[0], "p[0]");   // CHECK: p[0] is not present

#pragma omp target enter data map(present, alloc : s1)
  // At 5.2 the run completes past this point; at 6.0 the present check on the
  // absent pointee s1.p[0:10] fails here.
  // CHECK-52: done
  //
  // clang-format off
  // CHECK-60: message: device mapping required by 'present' map type modifier does not exist for host address 0x{{0*}}[[#HOST_ADDR]] ([[#SIZE]] bytes)
  // CHECK-60: fatal error 1: failure of target construct while offloading is mandatory
  // clang-format on

  fprintf(stderr, "done\n");
}
