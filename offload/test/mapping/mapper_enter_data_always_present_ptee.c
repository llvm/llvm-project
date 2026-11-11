// Same as mapper_target_update_present_ptee.c, but drives the mapper with a
// "target enter data map(always, present, to : s)" clause instead of a
// "target update to(present : s)" motion clause. Both invoke the mapper; this
// checks present propagation to the pointee is consistent across the two paths.

// Inbounds: the pointee region is fully present; the run succeeds at every
// OpenMP version.
// RUN: %libomptarget-compile-generic -fopenmp-version=52
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic
// RUN: %libomptarget-compile-generic -fopenmp-version=60
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic

// Out-of-bounds: s.p[0:20] extends beyond the mapped region x[0:10]. Because a
// map clause performs an actual mapping, requesting 20 elements when only 10
// are present is an error, so the run fails with an "explicit extension"
// diagnostic at every version.
// FIXME: at OpenMP 6.0 the present modifier is also propagated to the pointee,
// so the failure should additionally report the 'present' map-type-modifier
// check on s.p[0:20]. The extension diagnostic currently fires first.
// RUN: %libomptarget-compile-generic -fopenmp-version=52 -DOUT_OF_BOUNDS
// RUN: %libomptarget-run-fail-generic 2>&1 \
// RUN: | %fcheck-generic --check-prefix=CHECK-OOB
// RUN: %libomptarget-compile-generic -fopenmp-version=60 -DOUT_OF_BOUNDS
// RUN: %libomptarget-run-fail-generic 2>&1 \
// RUN: | %fcheck-generic --check-prefix=CHECK-OOB

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

  // CHECK-OOB: explicit extension not allowed
#pragma omp target data map(from : s.y, x)
  {
    f1();
  }

  // CHECK: 333 333
  printf("%d %d\n", x[0], s.y);
}
