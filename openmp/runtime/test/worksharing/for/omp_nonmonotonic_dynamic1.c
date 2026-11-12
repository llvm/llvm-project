// RUN: %libomp-compile
// RUN: env OMP_SCHEDULE=nonmonotonic:dynamic,10 %libomp-run

// The test checks iterations distribution for OMP 5.0 nonmonotonic OMP_SCHEDULE
// case #threads > #chunks (fallback to monotonic dynamic)

#include <stdio.h>
#include <omp.h>

#define ITERS 100
#define CHUNK 10
int err = 0;

int main(int argc, char **argv) {
  int i, ch, it[ITERS];
  omp_set_num_threads(16); // #threads is bigger than #chunks
#pragma omp parallel for schedule(runtime)
  for (i = 0; i < ITERS; ++i) {
    it[i] = omp_get_thread_num();
  }
  // check that each chunk executed by single thread
  for (ch = 0; ch < ITERS/CHUNK; ++ch) {
    int iter = ch * CHUNK;
    int nt = it[iter]; // thread number
    for (i = 1; i < CHUNK; ++i) {
#if _DEBUG
      printf("iter %d: (%d %d)\n", iter + i, nt, it[iter + i]);
#endif
      if (nt != it[iter + i]) {
        err++;
      }
    }
  }
  if (err > 0) {
    printf("Failed, err = %d\n", err);
    return 1;
  }
  printf("Passed\n");
  return 0;
}
