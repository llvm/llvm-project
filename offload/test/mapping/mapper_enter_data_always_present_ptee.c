// Same as mapper_target_update_present_ptee.c, but drives the mapper with a
// "target enter data map(always, present, to : s)" clause instead of a
// "target update to(present : s)" motion clause. Both invoke the mapper; this
// checks present propagation to the pointee is consistent across the two paths.
//
// FIXME: this test currently run-fails at every version/bounds combination.
// The mapper maps the struct member (s.y) with a combined entry whose size does
// not match the member's own storage, so the map clause aborts with an
// "explicit extension not allowed" error before the present modifier is ever
// considered. This is fixed once the mapper emits attach-style maps for pointer
// members (so the member and pointee occupy separate, correctly-sized entries).
//
// EXPECTED final state:
//   inbounds, 5.2 and 6.0: run succeeds, prints "333 333".
//   out-of-bounds (s.p[0:20] over the mapped x[0:10]):
//     5.2: succeeds (present is not applied to the pointee before 6.0).
//     6.0: run-fails; the failure should be the 'present' map-type-modifier
//          check on s.p[0:20] (an accompanying "explicit extension" message is
//          incidental -- for a map clause it is user error to map 20 elements
//          when only 10 are present).

// RUN: %libomptarget-compile-generic -fopenmp-version=52
// RUN: %libomptarget-run-fail-generic 2>&1 | %fcheck-generic
// RUN: %libomptarget-compile-generic -fopenmp-version=60
// RUN: %libomptarget-run-fail-generic 2>&1 | %fcheck-generic
// RUN: %libomptarget-compile-generic -fopenmp-version=52 -DOUT_OF_BOUNDS
// RUN: %libomptarget-run-fail-generic 2>&1 | %fcheck-generic
// RUN: %libomptarget-compile-generic -fopenmp-version=60 -DOUT_OF_BOUNDS
// RUN: %libomptarget-run-fail-generic 2>&1 | %fcheck-generic

#include <stdio.h>

int x[10];
typedef struct {
  int y;
  int *p;
} S;

#ifdef OUT_OF_BOUNDS
#pragma omp declare mapper(S s) map(s.y, s.p[0 : 20])
#else
#pragma omp declare mapper(S s) map(s.y, s.p[0 : 2])
#endif
S s;

void f1() {
#pragma omp target enter data map(always, present, to : s)

#pragma omp target data use_device_addr(s, x)
#pragma omp target has_device_addr(s, x)
  {
    s.y = s.y + 222;
    x[0] = x[0] + 222;
  }

#pragma omp target exit data map(release : s)
}

int main() {
  x[0] = 111;
  s.y = 111;
  s.p = &x[0];

  fprintf(stderr, "addr=%p, size=%zu\n", &s.p[0], 20 * sizeof(s.p[0]));

  // FIXME: the map clause aborts here with an "explicit extension not allowed"
  // error on the mapper's combined member entry, at every version. Fixed once
  // the mapper uses attach-style maps for pointer members.
  // CHECK: explicit extension not allowed
#pragma omp target data map(from : s.y, x)
  {
    f1();
  }

  printf("%d %d\n", x[0], s.y);
}
