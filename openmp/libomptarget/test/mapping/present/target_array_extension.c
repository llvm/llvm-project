// --------------------------------------------------
// Check extends before
// --------------------------------------------------

// RUN: %libomptarget-compile-aarch64-unknown-linux-gnu \
// RUN:   -fopenmp-version=51 -DEXTENDS=BEFORE
// RUN: %libomptarget-run-fail-aarch64-unknown-linux-gnu 2>&1 \
// RUN: | %fcheck-aarch64-unknown-linux-gnu

// RUN: %libomptarget-compile-powerpc64-ibm-linux-gnu \
// RUN:   -fopenmp-version=51 -DEXTENDS=BEFORE
// RUN: %libomptarget-run-fail-powerpc64-ibm-linux-gnu 2>&1 \
// RUN: | %fcheck-powerpc64-ibm-linux-gnu

// RUN: %libomptarget-compile-powerpc64le-ibm-linux-gnu \
// RUN:   -fopenmp-version=51 -DEXTENDS=BEFORE
// RUN: %libomptarget-run-fail-powerpc64le-ibm-linux-gnu 2>&1 \
// RUN: | %fcheck-powerpc64le-ibm-linux-gnu

// RUN: %libomptarget-compile-x86_64-pc-linux-gnu \
// RUN:   -fopenmp-version=51 -DEXTENDS=BEFORE
// RUN: %libomptarget-run-fail-x86_64-pc-linux-gnu 2>&1 \
// RUN: | %fcheck-x86_64-pc-linux-gnu

// --------------------------------------------------
// Check extends after
// --------------------------------------------------

// RUN: %libomptarget-compile-aarch64-unknown-linux-gnu \
// RUN:   -fopenmp-version=51 -DEXTENDS=AFTER
// RUN: %libomptarget-run-fail-aarch64-unknown-linux-gnu 2>&1 \
// RUN: | %fcheck-aarch64-unknown-linux-gnu

// RUN: %libomptarget-compile-powerpc64-ibm-linux-gnu \
// RUN:   -fopenmp-version=51 -DEXTENDS=AFTER
// RUN: %libomptarget-run-fail-powerpc64-ibm-linux-gnu 2>&1 \
// RUN: | %fcheck-powerpc64-ibm-linux-gnu

// RUN: %libomptarget-compile-powerpc64le-ibm-linux-gnu \
// RUN:   -fopenmp-version=51 -DEXTENDS=AFTER
// RUN: %libomptarget-run-fail-powerpc64le-ibm-linux-gnu 2>&1 \
// RUN: | %fcheck-powerpc64le-ibm-linux-gnu

// RUN: %libomptarget-compile-x86_64-pc-linux-gnu \
// RUN:   -fopenmp-version=51 -DEXTENDS=AFTER
// RUN: %libomptarget-run-fail-x86_64-pc-linux-gnu 2>&1 \
// RUN: | %fcheck-x86_64-pc-linux-gnu

// END.

#include <stdio.h>

#define BEFORE 0
#define AFTER  1

#define SIZE 100

#if EXTENDS == BEFORE
# define SMALL_BEG (SIZE-2)
# define SMALL_END SIZE
# define LARGE_BEG 0
# define LARGE_END SIZE
#elif EXTENDS == AFTER
# define SMALL_BEG 0
# define SMALL_END 2
# define LARGE_BEG 0
# define LARGE_END SIZE
#else
# error EXTENDS undefined
#endif

#define SMALL_SIZE (SMALL_END-SMALL_BEG)
#define LARGE_SIZE (LARGE_END-LARGE_BEG)

#define SMALL SMALL_BEG:SMALL_SIZE
#define LARGE LARGE_BEG:LARGE_SIZE

int main() {
  int arr[SIZE];

  // CHECK: addr=0x[[#%x,SMALL_ADDR:]], size=[[#%u,SMALL_BYTES:]]
  fprintf(stderr, "addr=%p, size=%ld\n", &arr[SMALL_BEG],
          SMALL_SIZE * sizeof arr[0]);

  // CHECK: addr=0x[[#%x,LARGE_ADDR:]], size=[[#%u,LARGE_BYTES:]]
  fprintf(stderr, "addr=%p, size=%ld\n", &arr[LARGE_BEG],
          LARGE_SIZE * sizeof arr[0]);

  // CHECK-NOT: Libomptarget
#pragma omp target data map(alloc: arr[LARGE])
  {
#pragma omp target map(present, tofrom: arr[SMALL])
    ;
  }

  // CHECK: arr is present
  fprintf(stderr, "arr is present\n");

  // CHECK: Libomptarget message: explicit extension not allowed: host address specified is 0x{{0*}}[[#LARGE_ADDR]] ([[#LARGE_BYTES]] bytes), but device allocation maps to host at 0x{{0*}}[[#SMALL_ADDR]] ([[#SMALL_BYTES]] bytes)
  // CHECK: Libomptarget message: device mapping required by 'present' map type modifier does not exist for host address 0x{{0*}}[[#LARGE_ADDR]] ([[#LARGE_BYTES]] bytes)
  // CHECK: Libomptarget fatal error 1: failure of target construct while offloading is mandatory
#pragma omp target data map(alloc: arr[SMALL])
  {
#pragma omp target map(present, tofrom: arr[LARGE])
    ;
  }

  // CHECK-NOT: arr is present
  fprintf(stderr, "arr is present\n");

  return 0;
}
