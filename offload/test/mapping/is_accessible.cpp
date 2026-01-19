// RUN: %libomptarget-compilexx-generic
// RUN: env HSA_XNACK=1 %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic

// RUN: %libomptarget-compilexx-generic
// RUN: env HSA_XNACK=0 %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefix=NO_USM

// REQUIRES: unified_shared_memory
// XFAIL: nvptx
// XFAIL: intelgpu

// CHECK: SUCCESS
// NO_USM: Not accessible

#include <assert.h>
#include <iostream>
#include <omp.h>
#include <stdio.h>

int main() {
  int n = 10000;
  int *a = new int[n];
  int err = 0;

  // program must be executed with HSA_XNACK=1
  if (!omp_target_is_accessible(a, n * sizeof(int), /*device_num=*/0))
    printf("Not accessible\n");
  else {
#pragma omp target teams distribute parallel for
    for (int i = 0; i < n; i++)
      a[i] = i;

    for (int i = 0; i < n; i++)
      if (a[i] != i)
        err++;
  }

  printf("%s\n", err == 0 ? "SUCCESS" : "FAIL");
  return err;
}
