// RUN: %libomptarget-compile-generic -fopenmp-extensions
// RUN: %libomptarget-run-generic | %fcheck-generic -strict-whitespace

#include <omp.h>
#include <stdio.h>

#define CHECK_PRESENCE(Var1, Var2, Var3)                                       \
  printf("    presence of %s, %s, %s: %d, %d, %d\n", #Var1, #Var2, #Var3,      \
         omp_target_is_present(&Var1, omp_get_default_device()),               \
         omp_target_is_present(&Var2, omp_get_default_device()),               \
         omp_target_is_present(&Var3, omp_get_default_device()))

int main() {
  int m, r, d;
  // CHECK: presence of m, r, d: 0, 0, 0
  CHECK_PRESENCE(m, r, d);

  // -----------------------------------------------------------------------
  // CHECK-NEXT: check:{{.*}}
  printf("check: dyn>0, hold=0, dec dyn=0\n");

  // CHECK-NEXT: once
  printf("  once\n");
#pragma omp target map(tofrom : m) map(alloc : r, d)
  ;
  // CHECK-NEXT: presence of m, r, d: 0, 0, 0
  CHECK_PRESENCE(m, r, d);

  // CHECK-NEXT: twice
  printf("  twice\n");
#pragma omp target data map(tofrom : m) map(alloc : r, d)
  {
    // CHECK-NEXT: presence of m, r, d: 1, 1, 1
    CHECK_PRESENCE(m, r, d);
#pragma omp target map(tofrom : m) map(alloc : r, d)
    ;
    // CHECK-NEXT: presence of m, r, d: 1, 1, 1
    CHECK_PRESENCE(m, r, d);
  }
  // CHECK-NEXT: presence of m, r, d: 0, 0, 0
  CHECK_PRESENCE(m, r, d);

  // -----------------------------------------------------------------------
  // CHECK: check:{{.*}}
  printf("check: dyn=0, hold>0, dec hold=0\n");

  // CHECK-NEXT: once
  printf("  once\n");
#pragma omp target map(ompx_hold, tofrom : m) map(ompx_hold, alloc : r, d)
  ;
  // CHECK-NEXT: presence of m, r, d: 0, 0, 0
  CHECK_PRESENCE(m, r, d);

  // CHECK-NEXT: twice
  printf("  twice\n");
#pragma omp target data map(ompx_hold, tofrom : m) map(ompx_hold, alloc : r, d)
  {
    // CHECK-NEXT: presence of m, r, d: 1, 1, 1
    CHECK_PRESENCE(m, r, d);
#pragma omp target map(ompx_hold, tofrom : m) map(ompx_hold, alloc : r, d)
    ;
    // CHECK-NEXT: presence of m, r, d: 1, 1, 1
    CHECK_PRESENCE(m, r, d);
  }
  // CHECK-NEXT: presence of m, r, d: 0, 0, 0
  CHECK_PRESENCE(m, r, d);

  // -----------------------------------------------------------------------
  // CHECK: check:{{.*}}
  printf("check: dyn>0, hold>0, dec dyn=0, dec hold=0\n");

  // CHECK-NEXT: once each
  printf("  once each\n");
#pragma omp target data map(ompx_hold, tofrom : m) map(ompx_hold, alloc : r, d)
  {
    // CHECK-NEXT: presence of m, r, d: 1, 1, 1
    CHECK_PRESENCE(m, r, d);
#pragma omp target map(tofrom : m) map(alloc : r, d)
    ;
    // CHECK-NEXT: presence of m, r, d: 1, 1, 1
    CHECK_PRESENCE(m, r, d);
  }
  // CHECK-NEXT: presence of m, r, d: 0, 0, 0
  CHECK_PRESENCE(m, r, d);

  // CHECK-NEXT: twice each
  printf("  twice each\n");
#pragma omp target data map(ompx_hold, tofrom : m) map(ompx_hold, alloc : r, d)
  {
    // CHECK-NEXT: presence of m, r, d: 1, 1, 1
    CHECK_PRESENCE(m, r, d);
#pragma omp target data map(ompx_hold, tofrom : m) map(ompx_hold, alloc : r, d)
    {
      // CHECK-NEXT: presence of m, r, d: 1, 1, 1
      CHECK_PRESENCE(m, r, d);
#pragma omp target data map(tofrom : m) map(alloc : r, d)
      {
        // CHECK-NEXT: presence of m, r, d: 1, 1, 1
        CHECK_PRESENCE(m, r, d);
#pragma omp target map(tofrom : m) map(alloc : r, d)
        ;
        // CHECK-NEXT: presence of m, r, d: 1, 1, 1
        CHECK_PRESENCE(m, r, d);
      }
      // CHECK-NEXT: presence of m, r, d: 1, 1, 1
      CHECK_PRESENCE(m, r, d);
    }
    // CHECK-NEXT: presence of m, r, d: 1, 1, 1
    CHECK_PRESENCE(m, r, d);
  }
  // CHECK-NEXT: presence of m, r, d: 0, 0, 0
  CHECK_PRESENCE(m, r, d);

  // -----------------------------------------------------------------------
  // CHECK: check:{{.*}}
  printf("check: dyn>0, hold>0, dec hold=0, dec dyn=0\n");

  // CHECK-NEXT: once each
  printf("  once each\n");
#pragma omp target data map(tofrom : m) map(alloc : r, d)
  {
    // CHECK-NEXT: presence of m, r, d: 1, 1, 1
    CHECK_PRESENCE(m, r, d);
#pragma omp target map(ompx_hold, tofrom : m) map(ompx_hold, alloc : r, d)
    ;
    // CHECK-NEXT: presence of m, r, d: 1, 1, 1
    CHECK_PRESENCE(m, r, d);
  }
  // CHECK-NEXT: presence of m, r, d: 0, 0, 0
  CHECK_PRESENCE(m, r, d);

  // CHECK-NEXT: twice each
  printf("  twice each\n");
#pragma omp target data map(tofrom : m) map(alloc : r, d)
  {
    // CHECK-NEXT: presence of m, r, d: 1, 1, 1
    CHECK_PRESENCE(m, r, d);
#pragma omp target data map(tofrom : m) map(alloc : r, d)
    {
      // CHECK-NEXT: presence of m, r, d: 1, 1, 1
      CHECK_PRESENCE(m, r, d);
#pragma omp target data map(ompx_hold, tofrom : m) map(ompx_hold, alloc : r, d)
      {
        // CHECK-NEXT: presence of m, r, d: 1, 1, 1
        CHECK_PRESENCE(m, r, d);
#pragma omp target map(ompx_hold, tofrom : m) map(ompx_hold, alloc : r, d)
        ;
        // CHECK-NEXT: presence of m, r, d: 1, 1, 1
        CHECK_PRESENCE(m, r, d);
      }
      // CHECK-NEXT: presence of m, r, d: 1, 1, 1
      CHECK_PRESENCE(m, r, d);
    }
    // CHECK-NEXT: presence of m, r, d: 1, 1, 1
    CHECK_PRESENCE(m, r, d);
  }
  // CHECK-NEXT: presence of m, r, d: 0, 0, 0
  CHECK_PRESENCE(m, r, d);

  return 0;
}
