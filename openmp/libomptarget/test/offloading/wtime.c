// RUN: %libomptarget-compileopt-run-and-check-generic

// UNSUPPORTED: amdgcn-amd-amdhsa
// UNSUPPORTED: amdgcn-amd-amdhsa-oldDriver
// UNSUPPORTED: amdgcn-amd-amdhsa-LTO

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N (1024 * 1024 * 256)

int main(int argc, char *argv[]) {
  int *data = (int *)malloc(N * sizeof(int));
#pragma omp target map(from: data[0:N])
  {
    double start = omp_get_wtime();
    for (int i = 0; i < N; ++i)
      data[i] = i;
    double end = omp_get_wtime();
    double duration = end - start;
    printf("duration: %lfs\n", duration);
  }
  free(data);
  return 0;
}

// CHECK: duration: {{.+[1-9]+}}

