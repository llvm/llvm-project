// RUN: %libomptarget-compile-aarch64-unknown-linux-gnu
// RUN: %libomptarget-run-aarch64-unknown-linux-gnu 2>&1 \
// RUN: | %fcheck-aarch64-unknown-linux-gnu

// RUN: %libomptarget-compile-powerpc64-ibm-linux-gnu
// RUN: %libomptarget-run-powerpc64-ibm-linux-gnu 2>&1 \
// RUN: | %fcheck-powerpc64-ibm-linux-gnu

// RUN: %libomptarget-compile-powerpc64le-ibm-linux-gnu
// RUN: %libomptarget-run-powerpc64le-ibm-linux-gnu 2>&1 \
// RUN: | %fcheck-powerpc64le-ibm-linux-gnu

// RUN: %libomptarget-compile-x86_64-pc-linux-gnu
// RUN: %libomptarget-run-x86_64-pc-linux-gnu 2>&1 \
// RUN: | %fcheck-x86_64-pc-linux-gnu
//
// END.

#include <omp.h>
#include <stdio.h>

int main() {
  int arr[100];

#pragma omp target data map(alloc: arr[50:2]) // partially mapped
  {
#pragma omp target // would implicitly map with full size but already present
    {
      arr[50] = 5;
      arr[51] = 6;
    } // must treat as present (dec ref count) even though full size not present
  } // wouldn't delete if previous ref count dec didn't happen

  // CHECK: still present: 0
  fprintf(stderr, "still present: %d\n",
          omp_target_is_present(&arr[50], omp_get_default_device()));

  return 0;
}
