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

#pragma omp target enter data map(to : A[FROM : LENGTH])

  // A, has been mapped starting at index FROM, but inside the use_device_ptr
  // clause it is captured by base so the library must look it up using the
  // base address.

  float *A_dev = NULL;
#pragma omp target data use_device_ptr(A)
  { A_dev = A; }
#pragma omp target exit data map(delete : A[FROM : LENGTH])

  // CHECK: Success
  if (A_dev == NULL || A_dev == A)
    fprintf(stderr, "Failure\n");
  else
    fprintf(stderr, "Success\n");

  free(A);

  return 0;
}
