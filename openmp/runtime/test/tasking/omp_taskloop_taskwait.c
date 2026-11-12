// RUN: %libomp-compile-and-run
#include <stdio.h>
#include <omp.h>
int main()
{
  enum {ITERS = 500};
  enum {SIZE = 5};
  int err = 0;
  #pragma omp parallel num_threads(2) reduction(+:err)
  {
    int r = 0;
    int i;
    #pragma omp taskloop grainsize(SIZE) shared(r) nogroup
    for(i=0; i<ITERS; i++) {
      #pragma omp atomic
        ++r;
    }
    #pragma omp taskwait
    printf("%d\n", r);
    if (r != ITERS)
      err++;
  } // end of parallel
  if (err != 0) {
    printf("failed, err = %d\n", err);
    return 1;
  } else {
    printf("passed\n");
    return 0;
  }
}
