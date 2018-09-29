// RUN: %compile-run-and-check

#include <omp.h>
#include <stdio.h>

const int MaxThreads = 1024;

int main(int argc, char *argv[]) {
  int cancellation = -1, dynamic = -1, nested = -1, maxActiveLevels = -1;

  #pragma omp target map(cancellation, dynamic, nested, maxActiveLevels)
  {
    // libomptarget-nvptx doesn't support cancellation.
    cancellation = omp_get_cancellation();

    // No support for dynamic adjustment of the number of threads.
    omp_set_dynamic(1);
    dynamic = omp_get_dynamic();

    // libomptarget-nvptx doesn't support nested parallelism.
    omp_set_nested(1);
    nested = omp_get_nested();

    omp_set_max_active_levels(42);
    maxActiveLevels = omp_get_max_active_levels();
  }

  // CHECK: cancellation = 0
  printf("cancellation = %d\n", cancellation);
  // CHECK: dynamic = 0
  printf("dynamic = %d\n", dynamic);
  // CHECK: nested = 0
  printf("nested = %d\n", nested);
  // CHECK: maxActiveLevels = 1
  printf("maxActiveLevels = %d\n", maxActiveLevels);

  return 0;
}
