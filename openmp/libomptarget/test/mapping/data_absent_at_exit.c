// RUN: %libomptarget-compile-run-and-check-aarch64-unknown-linux-gnu
// RUN: %libomptarget-compile-run-and-check-powerpc64-ibm-linux-gnu
// RUN: %libomptarget-compile-run-and-check-powerpc64le-ibm-linux-gnu
// RUN: %libomptarget-compile-run-and-check-x86_64-pc-linux-gnu
// RUN: %libomptarget-compile-run-and-check-nvptx64-nvidia-cuda

#include <stdio.h>

// OpenMP 5.1, sec. 2.21.7.1 "map Clause", p. 351 L14-16:
// "If the map clause appears on a target, target data, or target exit data
// construct and a corresponding list item of the original list item is not
// present in the device data environment on exit from the region then the
// list item is ignored."

int main(void) {
  int f = 5, r = 6, d = 7, af = 8;

  // Check exit from omp target data.
  // CHECK: f = 5, af = 8
  #pragma omp target data map(from: f) map(always, from: af)
  {
    #pragma omp target exit data map(delete: f, af)
  }
  printf("f = %d, af = %d\n", f, af);

  // Check omp target exit data.
  // CHECK: f = 5, r = 6, d = 7, af = 8
  #pragma omp target exit data map(from: f) map(release: r) map(delete: d) \
                               map(always, from: af)
  printf("f = %d, r = %d, d = %d, af = %d\n", f, r, d, af);

  return 0;
}
