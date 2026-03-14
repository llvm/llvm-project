// RUN: %libomp-compile && env LIBOMP_NUM_HIDDEN_HELPER_THREADS=1 %libomp-run

// gcc/icc target offloading is incompatible with libomp
// UNSUPPORTED: icc, gcc

// The test checks that "devide-by-0" bug fixed in runtime.
// The fix is to increment number of threads by 1 if positive,
// so that operation
//   (gtid) % (__kmp_hidden_helper_threads_num - 1)
// does not cause crash.

#include <stdio.h>
#include <omp.h>

int main(){
#pragma omp target nowait
   {
      printf("----- in  target region\n");
   }
  printf("------ before taskwait\n");
#pragma omp taskwait
  printf("passed\n");
  return 0;
}
