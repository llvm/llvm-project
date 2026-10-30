// RUN: %libomptarget-compile-generic
// RUN: env LIBOMPTARGET_INFO=32 %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic

// Same scenario as attach_ptr_update_to_restore.c, but checked through the
// data-transfer INFO output so it also runs without OMPTARGET_DEBUG. Logging
// the pointer restore must not re-lock the map entry the update already holds.

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
  // CHECK-NOT:   Size=8,
  fprintf(stderr, "update to(s.x)\n");
  s.x = 1;
#pragma omp target update to(s.x)

  // The pointer-sized copy after the struct copy is the pointer restore.
  // CHECK-LABEL: update to(s)
  // CHECK:       Copying data from host to device, {{.*}}Size=16,
  // CHECK:       Copying data from host to device, {{.*}}Size=8,
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
