// --------------------------------------------------
// Check 'to' and extends before
// --------------------------------------------------

// RUN: %libomptarget-compile-generic \
// RUN:   -DCLAUSE=to -DEXTENDS=BEFORE
// RUN: %libomptarget-run-fail-generic 2>&1 \
// RUN: | %fcheck-generic

// --------------------------------------------------
// Check 'from' and extends before
// --------------------------------------------------

// RUN: %libomptarget-compile-generic \
// RUN:   -DCLAUSE=from -DEXTENDS=BEFORE
// RUN: %libomptarget-run-fail-generic 2>&1 \
// RUN: | %fcheck-generic

// --------------------------------------------------
// Check 'to' and extends after
// --------------------------------------------------

// RUN: %libomptarget-compile-generic \
// RUN:   -DCLAUSE=to -DEXTENDS=AFTER
// RUN: %libomptarget-run-fail-generic 2>&1 \
// RUN: | %fcheck-generic

// --------------------------------------------------
// Check 'from' and extends after
// --------------------------------------------------

// RUN: %libomptarget-compile-generic \
// RUN:   -DCLAUSE=from -DEXTENDS=AFTER
// RUN: %libomptarget-run-fail-generic 2>&1 \
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

  // CHECK: addr=0x[[#%x,HOST_ADDR:]], size=[[#%u,SIZE:]]
  fprintf(stderr, "addr=%p, size=%ld\n", arr, sizeof arr);

  // CHECK-NOT: Libomptarget
#pragma omp target data map(alloc : arr[LARGE])
  {
#pragma omp target update CLAUSE(present : arr[SMALL])
  }

  // CHECK: arr is present
  fprintf(stderr, "arr is present\n");

  // CHECK: Libomptarget message: device mapping required by 'present' motion modifier does not exist for host address 0x{{0*}}[[#HOST_ADDR]] ([[#SIZE]] bytes)
  // CHECK: Libomptarget fatal error 1: failure of target construct while offloading is mandatory
#pragma omp target data map(alloc : arr[SMALL])
  {
#pragma omp target update CLAUSE(present : arr[LARGE])
  }

  // CHECK-NOT: arr is present
  fprintf(stderr, "arr is present\n");

  return 0;
}
