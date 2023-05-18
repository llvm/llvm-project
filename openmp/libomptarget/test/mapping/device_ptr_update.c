// RUN: %libomptarget-compile-generic
// RUN: env LIBOMPTARGET_DEBUG=1 %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefix=DEBUG -check-prefix=CHECK
// REQUIRES: libomptarget-debug

#include <stdio.h>

struct S {
  int *p;
};

int main(void) {
  int A[10];
  struct S s1;

  s1.p = A;

// DEBUG: Update pointer ([[DEV_PTR:0x[^ ]+]]) -> {{\[}}[[DEV_OBJ_A:0x[^ ]+]]{{\]}}
#pragma omp target enter data map(alloc : s1.p[0 : 10])

// DEBUG-NOT: Update pointer ([[DEV_PTR]]) -> {{\[}}[[DEV_OBJ_A]]{{\]}}
#pragma omp target map(alloc : s1.p[0 : 10])
  {
    for (int i = 0; i < 10; ++i)
      s1.p[i] = i;
  }

#pragma omp target exit data map(from : s1.p[0 : 10])

  int fail_A = 0;
  for (int i = 0; i < 10; ++i) {
    if (A[i] != i) {
      fail_A = 1;
      break;
    }
  }

  // CHECK-NOT: Test A failed
  if (fail_A) {
    printf("Test A failed\n");
  }

  return fail_A;
}
