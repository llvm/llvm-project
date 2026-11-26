// RUN: %libomptarget-compile-run-and-check-generic

#include <omp.h>
#include <stdio.h>

// OpenMP 5.1. sec 5.8.6 "Pointer Initialization for Device Data Environments"
// p. 160 L32-33: "If a matching mapped list item is not found, the pointer
// retains its original value as per the32 firstprivate semantics described in
// Section 5.4.4."

int main(void) {
  int *A = (int *)omp_target_alloc(sizeof(int), omp_get_default_device());

#pragma omp target
  { *A = 1; }

  int Result = 0;
#pragma omp target map(from : Result)
  { Result = *A; }

  // CHECK: PASS
  if (Result == 1)
    printf("PASS\n");

  omp_target_free(A, omp_get_default_device());
}
