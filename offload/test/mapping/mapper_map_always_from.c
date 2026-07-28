// Show that the ALWAYS map-type modifier on the outer map clause should be
// propagated to the entries pushed by a user-defined mapper.
//
// The mapper transfers s.y. We pre-map s.y so that on the target region below
// it already has a device copy with a nonzero reference count. Without ALWAYS,
// the `from` at the end of the target region is suppressed (a present,
// ref-counted entry is not copied back), so the write of 111 is lost. ALWAYS
// must force the copy-back -- but only if ALWAYS is propagated from the outer
// clause to the mapper's s.y entry.
//
// FIXME: ALWAYS is not propagated to the mapper's entries yet, so the copy-back
// is currently suppressed and s.y reads back as 0 (its pre-map value). Once
// ALWAYS is propagated:
//   EXPECTED: s.y = 111

// RUN: %libomptarget-compile-run-and-check-generic

#include <stdio.h>

typedef struct {
  int x;
  int y;
  int z;
} S;

#pragma omp declare mapper(default : S s) map(tofrom : s.y)

S s;

int main() {
  s.y = 0;

#pragma omp target enter data map(alloc : s.y)

#pragma omp target map(always, from : s)
  {
    s.y = 111;
  }

  // ALWAYS should force s.y back even though it is still mapped (ref count >
  // 0), but the modifier is not propagated yet, so the copy-back does not
  // happen.
  printf("s.y = %d\n", s.y); // CHECK: s.y = 0

#pragma omp target exit data map(delete : s.y)
}
