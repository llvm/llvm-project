// RUN: %compile-run-and-check

#include <stdio.h>
#include <omp.h>

const int WarpSize = 32;
const int ThreadLimit = 1 * WarpSize;
const int NumThreads2 = 2 * WarpSize;
const int NumThreads3 = 3 * WarpSize;
const int MaxThreads = 1024;

int main(int argc, char *argv[]) {
  int check1[MaxThreads];
  int check2[MaxThreads];
  int check3[MaxThreads];
  for (int i = 0; i < MaxThreads; i++) {
    check1[i] = check2[i] = check3[i] = 0;
  }

  int threadLimit = -1;

  #pragma omp target teams num_teams(1) thread_limit(ThreadLimit) \
                           map(check1[:], check2[:], check3[:], threadLimit)
  {
    threadLimit = omp_get_thread_limit();

    // All parallel regions should get as many threads as specified by the
    // thread_limit() clause.
    #pragma omp parallel
    {
      check1[omp_get_thread_num()] += omp_get_num_threads();
    }

    omp_set_num_threads(NumThreads2);
    #pragma omp parallel
    {
      check2[omp_get_thread_num()] += omp_get_num_threads();
    }

    #pragma omp parallel num_threads(NumThreads3)
    {
      check3[omp_get_thread_num()] += omp_get_num_threads();
    }
  }

  // CHECK: threadLimit = 32
  printf("threadLimit = %d\n", threadLimit);

  // CHECK-NOT: invalid
  for (int i = 0; i < MaxThreads; i++) {
    if (i < ThreadLimit) {
      if (check1[i] != ThreadLimit) {
        printf("invalid: check1[%d] should be %d, is %d\n", i, ThreadLimit, check1[i]);
      }
    } else if (check1[i] != 0) {
      printf("invalid: check1[%d] should be 0, is %d\n", i, check1[i]);
    }

    if (i < ThreadLimit) {
      if (check2[i] != ThreadLimit) {
        printf("invalid: check2[%d] should be %d, is %d\n", i, ThreadLimit, check2[i]);
      }
    } else if (check2[i] != 0) {
      printf("invalid: check2[%d] should be 0, is %d\n", i, check2[i]);
    }

    if (i < ThreadLimit) {
      if (check3[i] != ThreadLimit) {
        printf("invalid: check3[%d] should be %d, is %d\n", i, ThreadLimit, check3[i]);
      }
    } else if (check3[i] != 0) {
      printf("invalid: check3[%d] should be 0, is %d\n", i, check3[i]);
    }
  }

  return 0;
}
