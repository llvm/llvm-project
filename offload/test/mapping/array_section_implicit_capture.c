// RUN: %libomptarget-compile-generic 
// RUN: %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic

#include <stdio.h>
#include <stdlib.h>

#define N 1024
#define FROM 64
#define LENGTH 128

int main() {
  float *A = (float *)malloc(N * sizeof(float));
  float *B = (float *)malloc(N * sizeof(float));
  float *C = (float *)malloc(N * sizeof(float));

  for (int i = 0; i < N; i++) {
    C[i] = 0.0;
  }

  for (int i = 0; i < N; i++) {
    A[i] = i;
    B[i] = 2 * i;
  }

#pragma omp target enter data map(to : A[FROM : LENGTH], B[FROM : LENGTH])
#pragma omp target enter data map(alloc : C[FROM : LENGTH])

// A, B and C have been mapped starting at index FROM, but inside the kernel
// they are captured implicitly so the library must look them up using their
// base address.
#pragma omp target
  {
    for (int i = FROM; i < FROM + LENGTH; i++) {
      C[i] = A[i] + B[i];
    }
  }

#pragma omp target exit data map(from : C[FROM : LENGTH])
#pragma omp target exit data map(delete : A[FROM : LENGTH], B[FROM : LENGTH])

  int errors = 0;
  for (int i = FROM; i < FROM + LENGTH; i++)
    if (C[i] != A[i] + B[i])
      ++errors;

  // CHECK: Success
  if (errors)
    fprintf(stderr, "Failure\n");
  else
    fprintf(stderr, "Success\n");

  free(A);
  free(B);
  free(C);

  return 0;
}
