// RUN: %libomptarget-compileopt-run-and-check-generic

// UNSUPPORTED: amdgcn-amd-amdhsa
// UNSUPPORTED: amdgcn-amd-amdhsa-oldDriver
// UNSUPPORTED: amdgcn-amd-amdhsa-LTO

#include <omp.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  int data[1024];
#pragma omp target
  {
    double start = omp_get_wtime();
    for (int i = 0; i < 1024; ++i)
      data[i] = i;
    double end = omp_get_wtime();
    double duration = end - start;
    printf("duration: %lfs\n", duration);
  }
  return 0;
}

// CHECK: duration: {{.+[1-9]+}}
