// RUN: %libomptarget-compile-generic
// RUN: %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic

// END.

#include <omp.h>
#include <stdio.h>

int main() {
  int arr[100];

#pragma omp target data map(alloc : arr[50 : 2]) // partially mapped
  {
    // CHECK: arr[50] must present: 1
    fprintf(stderr, "arr[50] must present: %d\n",
            omp_target_is_present(&arr[50], omp_get_default_device()));

    // CHECK: arr[0] should not present: 0
    fprintf(stderr, "arr[0] should not present: %d\n",
            omp_target_is_present(&arr[0], omp_get_default_device()));

    // CHECK: arr[49] should not present: 0
    fprintf(stderr, "arr[49] should not present: %d\n",
            omp_target_is_present(&arr[49], omp_get_default_device()));

#pragma omp target // would implicitly map with full size but already present
    {
      arr[50] = 5;
      arr[51] = 6;
    } // must treat as present (dec ref count) even though full size not present
  }   // wouldn't delete if previous ref count dec didn't happen

  // CHECK: arr[50] still present: 0
  fprintf(stderr, "arr[50] still present: %d\n",
          omp_target_is_present(&arr[50], omp_get_default_device()));

  return 0;
}
