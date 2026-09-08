// Show that the ALWAYS map-type modifier on the outer map clause is propagated
// to the entries pushed by a user-defined mapper.
//
// The mapper transfers s.y. We pre-map s.y so that on the target region below
// it already has a device copy with a nonzero reference count. Without ALWAYS,
// the `from` at the end of the target region would be suppressed (a present,
// ref-counted entry is not copied back), so the write of 111 would be lost.
// ALWAYS forces the copy-back, which only happens if ALWAYS is propagated from
// the outer clause to the mapper's s.y entry.

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

  // ALWAYS forces s.y back even though it is still mapped (ref count > 0).
  printf("s.y = %d\n", s.y); // CHECK: s.y = 111

#pragma omp target exit data map(delete : s.y)
}
