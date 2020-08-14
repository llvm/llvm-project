// --------------------------------------------------
// Check 'to' and extends before
// --------------------------------------------------

// RUN: %libomptarget-compile-aarch64-unknown-linux-gnu \
// RUN:   -DCLAUSE=to -DEXTENDS=BEFORE
// RUN: %libomptarget-run-aarch64-unknown-linux-gnu 2>&1 \
// RUN: | %fcheck-aarch64-unknown-linux-gnu

// RUN: %libomptarget-compile-powerpc64-ibm-linux-gnu \
// RUN:   -DCLAUSE=to -DEXTENDS=BEFORE
// RUN: %libomptarget-run-powerpc64-ibm-linux-gnu 2>&1 \
// RUN: | %fcheck-powerpc64-ibm-linux-gnu

// RUN: %libomptarget-compile-powerpc64le-ibm-linux-gnu \
// RUN:   -DCLAUSE=to -DEXTENDS=BEFORE
// RUN: %libomptarget-run-powerpc64le-ibm-linux-gnu 2>&1 \
// RUN: | %fcheck-powerpc64le-ibm-linux-gnu

// RUN: %libomptarget-compile-x86_64-pc-linux-gnu \
// RUN:   -DCLAUSE=to -DEXTENDS=BEFORE
// RUN: %libomptarget-run-x86_64-pc-linux-gnu 2>&1 \
// RUN: | %fcheck-x86_64-pc-linux-gnu

// --------------------------------------------------
// Check 'from' and extends before
// --------------------------------------------------

// RUN: %libomptarget-compile-aarch64-unknown-linux-gnu \
// RUN:   -DCLAUSE=from -DEXTENDS=BEFORE
// RUN: %libomptarget-run-aarch64-unknown-linux-gnu 2>&1 \
// RUN: | %fcheck-aarch64-unknown-linux-gnu

// RUN: %libomptarget-compile-powerpc64-ibm-linux-gnu \
// RUN:   -DCLAUSE=from -DEXTENDS=BEFORE
// RUN: %libomptarget-run-powerpc64-ibm-linux-gnu 2>&1 \
// RUN: | %fcheck-powerpc64-ibm-linux-gnu

// RUN: %libomptarget-compile-powerpc64le-ibm-linux-gnu \
// RUN:   -DCLAUSE=from -DEXTENDS=BEFORE
// RUN: %libomptarget-run-powerpc64le-ibm-linux-gnu 2>&1 \
// RUN: | %fcheck-powerpc64le-ibm-linux-gnu

// RUN: %libomptarget-compile-x86_64-pc-linux-gnu \
// RUN:   -DCLAUSE=from -DEXTENDS=BEFORE
// RUN: %libomptarget-run-x86_64-pc-linux-gnu 2>&1 \
// RUN: | %fcheck-x86_64-pc-linux-gnu

// --------------------------------------------------
// Check 'to' and extends after
// --------------------------------------------------

// RUN: %libomptarget-compile-aarch64-unknown-linux-gnu \
// RUN:   -DCLAUSE=to -DEXTENDS=AFTER
// RUN: %libomptarget-run-aarch64-unknown-linux-gnu 2>&1 \
// RUN: | %fcheck-aarch64-unknown-linux-gnu

// RUN: %libomptarget-compile-powerpc64-ibm-linux-gnu \
// RUN:   -DCLAUSE=to -DEXTENDS=AFTER
// RUN: %libomptarget-run-powerpc64-ibm-linux-gnu 2>&1 \
// RUN: | %fcheck-powerpc64-ibm-linux-gnu

// RUN: %libomptarget-compile-powerpc64le-ibm-linux-gnu \
// RUN:   -DCLAUSE=to -DEXTENDS=AFTER
// RUN: %libomptarget-run-powerpc64le-ibm-linux-gnu 2>&1 \
// RUN: | %fcheck-powerpc64le-ibm-linux-gnu

// RUN: %libomptarget-compile-x86_64-pc-linux-gnu \
// RUN:   -DCLAUSE=to -DEXTENDS=AFTER
// RUN: %libomptarget-run-x86_64-pc-linux-gnu 2>&1 \
// RUN: | %fcheck-x86_64-pc-linux-gnu

// --------------------------------------------------
// Check 'from' and extends after
// --------------------------------------------------

// RUN: %libomptarget-compile-aarch64-unknown-linux-gnu \
// RUN:   -DCLAUSE=from -DEXTENDS=AFTER
// RUN: %libomptarget-run-aarch64-unknown-linux-gnu 2>&1 \
// RUN: | %fcheck-aarch64-unknown-linux-gnu

// RUN: %libomptarget-compile-powerpc64-ibm-linux-gnu \
// RUN:   -DCLAUSE=from -DEXTENDS=AFTER
// RUN: %libomptarget-run-powerpc64-ibm-linux-gnu 2>&1 \
// RUN: | %fcheck-powerpc64-ibm-linux-gnu

// RUN: %libomptarget-compile-powerpc64le-ibm-linux-gnu \
// RUN:   -DCLAUSE=from -DEXTENDS=AFTER
// RUN: %libomptarget-run-powerpc64le-ibm-linux-gnu 2>&1 \
// RUN: | %fcheck-powerpc64le-ibm-linux-gnu

// RUN: %libomptarget-compile-x86_64-pc-linux-gnu \
// RUN:   -DCLAUSE=from -DEXTENDS=AFTER
// RUN: %libomptarget-run-x86_64-pc-linux-gnu 2>&1 \
// RUN: | %fcheck-x86_64-pc-linux-gnu

// END.

#include <stdio.h>

#define BEFORE 0
#define AFTER  1

#if EXTENDS == BEFORE
# define SMALL 2:3
# define LARGE 0:5
#elif EXTENDS == AFTER
# define SMALL 0:3
# define LARGE 0:5
#else
# error EXTENDS undefined
#endif

int main() {
  int arr[5];

  // CHECK-NOT: Libomptarget
#pragma omp target data map(alloc: arr[LARGE])
  {
#pragma omp target update CLAUSE(arr[SMALL])
  }

  // CHECK: success
  fprintf(stderr, "success\n");

  // CHECK-NOT: Libomptarget
#pragma omp target data map(alloc: arr[SMALL])
  {
#pragma omp target update CLAUSE(arr[LARGE])
  }

  // CHECK: success
  fprintf(stderr, "success\n");

  return 0;
}
