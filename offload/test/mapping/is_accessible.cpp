// RUN: %libomptarget-compilexx-generic
// RUN: %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic

// REQUIRES: unified_shared_memory

#include <assert.h>
#include <iostream>
#include <omp.h>
#include <stdio.h>

// The runtime considers unified shared memory to be always present.
#pragma omp requires unified_shared_memory

int main() {
  int size = 10;
  int *x = (int *)malloc(size * sizeof(int));
  const int dev_num = omp_get_default_device();

  int is_accessible = omp_target_is_accessible(x, size * sizeof(int), dev_num);
  int errors = 0;
  int uses_shared_memory = 0;

#pragma omp target map(to : uses_shared_memory)
  uses_shared_memory = 1;

  assert(uses_shared_memory != is_accessible);

  if (is_accessible) {
#pragma omp target firstprivate(x)
    for (int i = 0; i < size; i++)
      x[i] = i * 3;

    for (int i = 0; i < size; i++)
      errors += (x[i] == (i * 3) ? 1 : 0);
  }

  free(x);
  // CHECK: x overwritten 0 times
  printf("x overwritten %d times\n", errors);

  return errors;
}
