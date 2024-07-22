// --------------------------------------------------
// Check 'to' and extends before
// --------------------------------------------------

// RUN: %libomptarget-compile-generic \
// RUN:   -DCLAUSE=to -DEXTENDS=BEFORE
// RUN: %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic

// --------------------------------------------------
// Check 'from' and extends before
// --------------------------------------------------

// RUN: %libomptarget-compile-generic \
// RUN:   -DCLAUSE=from -DEXTENDS=BEFORE
// RUN: %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic

// --------------------------------------------------
// Check 'to' and extends after
// --------------------------------------------------

// RUN: %libomptarget-compile-generic \
// RUN:   -DCLAUSE=to -DEXTENDS=AFTER
// RUN: %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic

// --------------------------------------------------
// Check 'from' and extends after
// --------------------------------------------------

// RUN: %libomptarget-compile-generic \
// RUN:   -DCLAUSE=from -DEXTENDS=AFTER
// RUN: %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic

// END.

#include <stdio.h>

#define BEFORE 0
#define AFTER 1

#if EXTENDS == BEFORE
#define SMALL 2 : 3
#define LARGE 0 : 5
#elif EXTENDS == AFTER
#define SMALL 0 : 3
#define LARGE 0 : 5
#else
#error EXTENDS undefined
#endif

int main() {
  int arr[5];

  // CHECK-NOT: omptarget
#pragma omp target data map(alloc : arr[LARGE])
  {
#pragma omp target update CLAUSE(arr[SMALL])
  }

  // CHECK: success
  fprintf(stderr, "success\n");

  // CHECK-NOT: omptarget
#pragma omp target data map(alloc : arr[SMALL])
  {
#pragma omp target update CLAUSE(arr[LARGE])
  }

  // CHECK: success
  fprintf(stderr, "success\n");

  return 0;
}
