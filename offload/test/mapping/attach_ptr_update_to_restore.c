// RUN: %libomptarget-compile-generic
// RUN: env LIBOMPTARGET_DEBUG=1 %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic
// REQUIRES: libomptarget-debug

// A target update to(...) overwrites an attached device pointer only if its
// range covers the pointer, and only then must the attachment be restored.

#include <stdio.h>

typedef struct {
  int *p;
  int x;
} S;

int a[1] = {7};

int main() {
  S s = {a, 0};
#pragma omp target enter data map(to : s, s.p[0 : 1])

  // CHECK-LABEL: update to(s.x)
  // CHECK-NOT:   Restoring target pointer
  fprintf(stderr, "update to(s.x)\n");
  s.x = 1;
#pragma omp target update to(s.x)

  // CHECK-LABEL: update to(s)
  // CHECK:       Restoring target pointer
  fprintf(stderr, "update to(s)\n");
  s.x = 2;
#pragma omp target update to(s)

  // CHECK-LABEL: done
  fprintf(stderr, "done\n");
  int p = -1, x = -1;
#pragma omp target map(from : p, x)
  {
    p = *s.p;
    x = s.x;
  }
  // CHECK: *s.p = 7, s.x = 2
  printf("*s.p = %d, s.x = %d\n", p, x);

#pragma omp target exit data map(delete : s) map(release : s.p[0 : 1])
  return 0;
}
