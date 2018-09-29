// RUN: %compile-run-and-check

#include <stdio.h>
#include <omp.h>

const int WarpSize = 32;
const int NumThreads1 = 1 * WarpSize;
const int NumThreads2 = 2 * WarpSize;
const int NumThreads3 = 3 * WarpSize;
const int MaxThreads = 1024;

int main(int argc, char *argv[]) {
  int check1[MaxThreads];
  int check2[MaxThreads];
  int check3[MaxThreads];
  int check4[MaxThreads];
  for (int i = 0; i < MaxThreads; i++) {
    check1[i] = check2[i] = check3[i] = check4[i] = 0;
  }

  int maxThreads1 = -1;
  int maxThreads2 = -1;
  int maxThreads3 = -1;

  #pragma omp target map(check1[:], check2[:], check3[:], check4[:]) \
                     map(maxThreads1, maxThreads2, maxThreads3)
  {
    #pragma omp parallel num_threads(NumThreads1)
    {
      check1[omp_get_thread_num()] += omp_get_num_threads();
    }

    // API method to set number of threads in parallel regions without
    // num_threads() clause.
    omp_set_num_threads(NumThreads2);
    maxThreads1 = omp_get_max_threads();
    #pragma omp parallel
    {
      check2[omp_get_thread_num()] += omp_get_num_threads();
    }

    maxThreads2 = omp_get_max_threads();

    // num_threads() clause should override nthreads-var ICV.
    #pragma omp parallel num_threads(NumThreads3)
    {
      check3[omp_get_thread_num()] += omp_get_num_threads();
    }

    maxThreads3 = omp_get_max_threads();

    // Effect from omp_set_num_threads() should still be visible.
    #pragma omp parallel
    {
      check4[omp_get_thread_num()] += omp_get_num_threads();
    }
  }

  // CHECK: maxThreads1 = 64
  printf("maxThreads1 = %d\n", maxThreads1);
  // CHECK: maxThreads2 = 64
  printf("maxThreads2 = %d\n", maxThreads2);
  // CHECK: maxThreads3 = 64
  printf("maxThreads3 = %d\n", maxThreads3);

  // CHECK-NOT: invalid
  for (int i = 0; i < MaxThreads; i++) {
    if (i < NumThreads1) {
      if (check1[i] != NumThreads1) {
        printf("invalid: check1[%d] should be %d, is %d\n", i, NumThreads1, check1[i]);
      }
    } else if (check1[i] != 0) {
      printf("invalid: check1[%d] should be 0, is %d\n", i, check1[i]);
    }

    if (i < NumThreads2) {
      if (check2[i] != NumThreads2) {
        printf("invalid: check2[%d] should be %d, is %d\n", i, NumThreads2, check2[i]);
      }
    } else if (check2[i] != 0) {
      printf("invalid: check2[%d] should be 0, is %d\n", i, check2[i]);
    }

    if (i < NumThreads3) {
      if (check3[i] != NumThreads3) {
        printf("invalid: check3[%d] should be %d, is %d\n", i, NumThreads3, check3[i]);
      }
    } else if (check3[i] != 0) {
      printf("invalid: check3[%d] should be 0, is %d\n", i, check3[i]);
    }

    if (i < NumThreads2) {
      if (check4[i] != NumThreads2) {
        printf("invalid: check4[%d] should be %d, is %d\n", i, NumThreads2, check4[i]);
      }
    } else if (check4[i] != 0) {
      printf("invalid: check4[%d] should be 0, is %d\n", i, check4[i]);
    }
  }

  return 0;
}
