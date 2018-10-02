// RUN: %compile-run-and-check

#include <omp.h>
#include <stdio.h>

const int MaxThreads = 1024;
const int NumThreads = 64;

int main(int argc, char *argv[]) {
  int inParallel = -1, numThreads = -1, threadNum = -1;
  int check1[MaxThreads];
  int check2[MaxThreads];
  for (int i = 0; i < MaxThreads; i++) {
    check1[i] = check2[i] = 0;
  }

  #pragma omp target map(inParallel, numThreads, threadNum, check1[:], check2[:])
  {
    inParallel = omp_in_parallel();
    numThreads = omp_get_num_threads();
    threadNum = omp_get_thread_num();

    // Expecting active parallel region.
    #pragma omp parallel num_threads(NumThreads)
    {
      int id = omp_get_thread_num();
      check1[id] += omp_get_num_threads() + omp_in_parallel();

      // Expecting serialized parallel region.
      #pragma omp parallel
      {
        // Expected to be 1.
        int nestedInParallel = omp_in_parallel();
        // Expected to be 1.
        int nestedNumThreads = omp_get_num_threads();
        // Expected to be 0.
        int nestedThreadNum = omp_get_thread_num();
        #pragma omp atomic
        check2[id] += nestedInParallel + nestedNumThreads + nestedThreadNum;
      }
    }
  }

  // CHECK: target: inParallel = 0, numThreads = 1, threadNum = 0
  printf("target: inParallel = %d, numThreads = %d, threadNum = %d\n",
         inParallel, numThreads, threadNum);

  // CHECK-NOT: invalid
  for (int i = 0; i < MaxThreads; i++) {
    // Check that all threads reported
    // omp_get_num_threads() = 64, omp_in_parallel() = 1.
    int Expected = NumThreads + 1;
    if (i < NumThreads) {
      if (check1[i] != Expected) {
        printf("invalid: check1[%d] should be %d, is %d\n", i, Expected, check1[i]);
      }
    } else if (check1[i] != 0) {
      printf("invalid: check1[%d] should be 0, is %d\n", i, check1[i]);
    }

    // Check serialized parallel region.
    if (i < NumThreads) {
      if (check2[i] != 2) {
        printf("invalid: check2[%d] should be 2, is %d\n", i, check2[i]);
      }
    } else if (check2[i] != 0) {
      printf("invalid: check2[%d] should be 0, is %d\n", i, check2[i]);
    }
  }

  return 0;
}
