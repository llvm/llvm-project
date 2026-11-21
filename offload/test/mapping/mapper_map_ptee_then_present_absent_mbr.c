// RUN: %libomptarget-compile-run-and-check-generic

#include <omp.h>
#include <stdio.h>

// A user-defined mapper maps a struct member (s.x) and a pointee (s.p[0:10]).
// We pre-map ONLY the pointee, leaving s.x absent on the device, then do a
// map(present: s) that invokes the mapper.
//
// FIXME: This currently PASSES (no present-check failure) even though s.x is
// absent. The mapper emits the s.x member component WITHOUT the PRESENT bit, so
// its absence is a silent no-op. PRESENT should be propagated to the s.x member
// entry pushed by the mapper, so that an absent member triggers the present
// check, but it currently is not.
//
// EXPECTED: the present check should FAIL (assert / abort) because s.x is not
// present on the device.

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

  // Pre-map ONLY the pointee s1.p[0:10], NOT s1.x.
#pragma omp target enter data map(alloc : s1.p[0 : 10])
  printf("After mapping\n");
  print_status(&s1.x, "x");       // CHECK: x is not present
  print_status(&s1.p[0], "p[0]"); // CHECK: p[0] is present
  printf("\n");

  // s1.x is NOT present on the device.
  // FIXME: this should trigger a present-check failure on s1.x, but currently
  // passes because PRESENT is not propagated to the s.x member entry.
  //   EXPECTED: run-fail with a "present" motion-modifier error for s1.x.
#pragma omp target enter data map(present, alloc : s1)
  printf("present check passed\n"); // CHECK: present check passed

#pragma omp target exit data map(delete : s1)
  return 0;
}
