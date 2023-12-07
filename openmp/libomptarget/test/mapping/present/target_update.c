// --------------------------------------------------
// Check 'to'
// --------------------------------------------------

// RUN: %libomptarget-compile-generic \
// RUN:   -DCLAUSE=to
// RUN: %libomptarget-run-fail-generic 2>&1 \
// RUN: | %fcheck-generic

// --------------------------------------------------
// Check 'from'
// --------------------------------------------------

// RUN: %libomptarget-compile-generic \
// RUN:   -DCLAUSE=from
// RUN: %libomptarget-run-fail-generic 2>&1 \
// RUN: | %fcheck-generic

#include <stdio.h>

int main() {
  int i;

  // CHECK: addr=0x[[#%x,HOST_ADDR:]], size=[[#%u,SIZE:]]
  fprintf(stderr, "addr=%p, size=%ld\n", &i, sizeof i);

  // CHECK-NOT: Libomptarget
#pragma omp target enter data map(alloc : i)
#pragma omp target update CLAUSE(present : i)
#pragma omp target exit data map(delete : i)

  // CHECK: i is present
  fprintf(stderr, "i is present\n");

  // CHECK: Libomptarget message: device mapping required by 'present' motion modifier does not exist for host address 0x{{0*}}[[#HOST_ADDR]] ([[#SIZE]] bytes)
  // CHECK: Libomptarget fatal error 1: failure of target construct while offloading is mandatory
#pragma omp target update CLAUSE(present : i)

  // CHECK-NOT: i is present
  fprintf(stderr, "i is present\n");

  return 0;
}
