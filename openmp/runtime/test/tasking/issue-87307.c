// RUN: %libomp-compile-and-run
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int a;

void inc_a() {
#pragma omp task
  {
#pragma omp atomic
    a++;
  }
}

int main() {
  int n;
  int nth_outer;
  omp_set_max_active_levels(2);
  omp_set_dynamic(0);

  for (n = 0; n < 200; ++n) {
    a = 0;
#pragma omp parallel num_threads(8)
    {
      if (omp_get_thread_num() == 0)
        nth_outer = omp_get_num_threads();
#pragma omp parallel num_threads(2)
      {
        int i;
#pragma omp master
        for (i = 0; i < 50; ++i)
          inc_a();
      }
    }
    if (a != nth_outer * 50) {
      fprintf(stderr, "error: a (%d) != %d\n", a, nth_outer * 50);
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}
