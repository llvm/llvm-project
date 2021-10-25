// RUN: %libomp-c99-compile-and-run
//
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define TYPE long
#define MAX_ITER (TYPE)((TYPE)1000000)
#define EVERY (TYPE)((TYPE)100000)

int main(int argc, char* argv[]) {
  TYPE x = MAX_ITER;
  omp_set_max_active_levels(2);
  omp_set_num_threads(2);
  #pragma omp parallel for schedule(nonmonotonic:dynamic,1)
  for (TYPE i = 0; i < x; i++) {
    int tid = omp_get_thread_num();
    omp_set_num_threads(1);
    #pragma omp parallel proc_bind(spread)
    {
      if (i % EVERY == (TYPE)0)
        printf("Outer thread %d at iter %ld\n", tid, i);
    }
  }
  printf("passed\n");
  return 0;
}
