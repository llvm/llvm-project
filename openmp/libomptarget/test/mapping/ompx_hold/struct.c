// RUN: %libomptarget-compile-generic -fopenmp-extensions
// RUN: %libomptarget-run-generic | %fcheck-generic -strict-whitespace

#include <omp.h>
#include <stdio.h>

#pragma omp begin declare target
char *N1, *N2;
int V1, V2;
#pragma omp declare target

#define CHECK_PRESENCE(Var1, Var2, Var3)                                       \
  printf("    presence of %s, %s, %s: %d, %d, %d\n", #Var1, #Var2, #Var3,      \
         omp_target_is_present(&(Var1), omp_get_default_device()),             \
         omp_target_is_present(&(Var2), omp_get_default_device()),             \
         omp_target_is_present(&(Var3), omp_get_default_device()))

#define CHECK_VALUES_HELPER(N1, N2, Var1, Var2)                                \
  printf("    values of %s, %s: %d, %d\n", N1, N2, (Var1), (Var2))

#define CHECK_VALUES_DELAYED(Var1, Var2)                                       \
  N1 = #Var1;                                                                  \
  N2 = #Var2;                                                                  \
  V1 = (Var1);                                                                 \
  V2 = (Var2);

#define CHECK_DELAYED_VALUS()                                                  \
  _Pragma("omp target update from(N1, N2, V1, V2)")                            \
      CHECK_VALUES_HELPER(N1, N2, V1, V2)

#define CHECK_VALUES(Var1, Var2)                                               \
  CHECK_VALUES_HELPER(#Var1, #Var2, (Var1), (Var2))

int main() {
  struct S {
    int i;
    int j;
  } s;
  // CHECK: presence of s, s.i, s.j: 0, 0, 0
  CHECK_PRESENCE(s, s.i, s.j);

  // =======================================================================
  // Check that ompx_hold keeps entire struct present.

  // -----------------------------------------------------------------------
  // CHECK-LABEL: check:{{.*}}
  printf("check: ompx_hold only on first member\n");
  s.i = 20;
  s.j = 30;
#pragma omp target data map(tofrom : s) map(ompx_hold, tofrom : s.i)           \
    map(tofrom : s.j)
  {
    // CHECK-NEXT: presence of s, s.i, s.j: 1, 1, 1
    CHECK_PRESENCE(s, s.i, s.j);
#pragma omp target map(tofrom : s)
    {
      s.i = 21;
      s.j = 31;
    }
#pragma omp target exit data map(delete : s, s.i)
    // ompx_hold on s.i applies to all of s.
    // CHECK-NEXT: presence of s, s.i, s.j: 1, 1, 1
    // CHECK-NEXT: values of s.i, s.j: 20, 30
    CHECK_PRESENCE(s, s.i, s.j);
    CHECK_VALUES(s.i, s.j);
  }
  // CHECK-NEXT: presence of s, s.i, s.j: 0, 0, 0
  // CHECK-NEXT: values of s.i, s.j: 21, 31
  CHECK_PRESENCE(s, s.i, s.j);
  CHECK_VALUES(s.i, s.j);

  // -----------------------------------------------------------------------
  // CHECK-LABEL: check:{{.*}}
  printf("check: ompx_hold only on last member\n");
  s.i = 20;
  s.j = 30;
#pragma omp target data map(tofrom : s) map(tofrom : s.i)                      \
    map(ompx_hold, tofrom : s.j)
  {
    // CHECK-NEXT: presence of s, s.i, s.j: 1, 1, 1
    CHECK_PRESENCE(s, s.i, s.j);
#pragma omp target map(tofrom : s)
    {
      s.i = 21;
      s.j = 31;
    }
#pragma omp target exit data map(delete : s, s.i)
    // ompx_hold on s.j applies to all of s.
    // CHECK-NEXT: presence of s, s.i, s.j: 1, 1, 1
    // CHECK-NEXT: values of s.i, s.j: 20, 30
    CHECK_PRESENCE(s, s.i, s.j);
    CHECK_VALUES(s.i, s.j);
  }
  // CHECK-NEXT: presence of s, s.i, s.j: 0, 0, 0
  // CHECK-NEXT: values of s.i, s.j: 21, 31
  CHECK_PRESENCE(s, s.i, s.j);
  CHECK_VALUES(s.i, s.j);

  // -----------------------------------------------------------------------
  // CHECK-LABEL: check:{{.*}}
  printf("check: ompx_hold only on struct\n");
  s.i = 20;
  s.j = 30;
#pragma omp target data map(ompx_hold, tofrom : s) map(tofrom : s.i)           \
    map(tofrom : s.j)
  {
    // CHECK-NEXT: presence of s, s.i, s.j: 1, 1, 1
    CHECK_PRESENCE(s, s.i, s.j);
#pragma omp target map(tofrom : s)
    {
      s.i = 21;
      s.j = 31;
    }
#pragma omp target exit data map(delete : s, s.i)
    // CHECK-NEXT: presence of s, s.i, s.j: 1, 1, 1
    // CHECK-NEXT: values of s.i, s.j: 20, 30
    CHECK_PRESENCE(s, s.i, s.j);
    CHECK_VALUES(s.i, s.j);
  }
  // CHECK-NEXT: presence of s, s.i, s.j: 0, 0, 0
  // CHECK-NEXT: values of s.i, s.j: 21, 31
  CHECK_PRESENCE(s, s.i, s.j);
  CHECK_VALUES(s.i, s.j);

  // =======================================================================
  // Check that transfer to/from host checks reference count correctly.

  // -----------------------------------------------------------------------
  // CHECK-LABEL: check:{{.*}}
  printf("check: parent DynRefCount=1 is not sufficient for transfer\n");
  s.i = 20;
  s.j = 30;
#pragma omp target data map(ompx_hold, tofrom : s)
#pragma omp target data map(ompx_hold, tofrom : s)
  {
    // CHECK-NEXT: presence of s, s.i, s.j: 1, 1, 1
    CHECK_PRESENCE(s, s.i, s.j);
#pragma omp target map(from : s.i, s.j)
    {
      s.i = 21;
      s.j = 31;
    } // No transfer here even though parent's DynRefCount=1.
    // CHECK-NEXT: presence of s, s.i, s.j: 1, 1, 1
    // CHECK-NEXT: values of s.i, s.j: 20, 30
    CHECK_PRESENCE(s, s.i, s.j);
    CHECK_VALUES(s.i, s.j);
#pragma omp target map(to : s.i, s.j)
    { // No transfer here even though parent's DynRefCount=1.
      // CHECK-NEXT: values of s.i, s.j: 21, 31
      CHECK_VALUES_DELAYED(s.i, s.j);
    }
    CHECK_DELAYED_VALUS();
  }
  // CHECK-NEXT: presence of s, s.i, s.j: 0, 0, 0
  // CHECK-NEXT: values of s.i, s.j: 21, 31
  CHECK_PRESENCE(s, s.i, s.j);
  CHECK_VALUES(s.i, s.j);

  // -----------------------------------------------------------------------
  // CHECK-LABEL: check:{{.*}}
  printf("check: parent HoldRefCount=1 is not sufficient for transfer\n");
  s.i = 20;
  s.j = 30;
#pragma omp target data map(tofrom : s)
#pragma omp target data map(tofrom : s)
  {
    // CHECK-NEXT: presence of s, s.i, s.j: 1, 1, 1
    CHECK_PRESENCE(s, s.i, s.j);
#pragma omp target map(ompx_hold, from : s.i, s.j)
    {
      s.i = 21;
      s.j = 31;
    } // No transfer here even though parent's HoldRefCount=1.
    // CHECK-NEXT: presence of s, s.i, s.j: 1, 1, 1
    // CHECK-NEXT: values of s.i, s.j: 20, 30
    CHECK_PRESENCE(s, s.i, s.j);
    CHECK_VALUES(s.i, s.j);
#pragma omp target map(ompx_hold, to : s.i, s.j)
    { // No transfer here even though parent's HoldRefCount=1.
      // CHECK-NEXT: values of s.i, s.j: 21, 31
      CHECK_VALUES_DELAYED(s.i, s.j);
    }
    CHECK_DELAYED_VALUS();
  }
  // CHECK-NEXT: presence of s, s.i, s.j: 0, 0, 0
  // CHECK-NEXT: values of s.i, s.j: 21, 31
  CHECK_PRESENCE(s, s.i, s.j);
  CHECK_VALUES(s.i, s.j);

  // -----------------------------------------------------------------------
  // CHECK-LABEL: check:{{.*}}
  //
  // At the beginning of a region, if the parent's TotalRefCount=1, then the
  // transfer should happen.
  //
  // At the end of a region, it also must be true that the reference count being
  // decremented is the reference count that is 1.
  printf("check: parent TotalRefCount=1 is not sufficient for transfer\n");
  s.i = 20;
  s.j = 30;
#pragma omp target data map(ompx_hold, tofrom : s)
  {
    // CHECK-NEXT: presence of s, s.i, s.j: 1, 1, 1
    CHECK_PRESENCE(s, s.i, s.j);
#pragma omp target map(ompx_hold, tofrom : s.i, s.j)
    {
      s.i = 21;
      s.j = 31;
    }
#pragma omp target exit data map(from : s.i, s.j)
    // No transfer here even though parent's TotalRefCount=1.
    // CHECK-NEXT: presence of s, s.i, s.j: 1, 1, 1
    // CHECK-NEXT: values of s.i, s.j: 20, 30
    CHECK_PRESENCE(s, s.i, s.j);
    CHECK_VALUES(s.i, s.j);
  }
  // CHECK-NEXT: presence of s, s.i, s.j: 0, 0, 0
  // CHECK-NEXT: values of s.i, s.j: 21, 31
  CHECK_PRESENCE(s, s.i, s.j);
  CHECK_VALUES(s.i, s.j);

  return 0;
}
