// RUN: %libomp-compile-and-run

// UNSUPPORTED: gcc-4, gcc-5, gcc-6, gcc-7, gcc-8

#include <stdio.h>
#include <stdlib.h>

int a = 0, b = 1;

int main(int argc, char **argv) {

  #pragma omp parallel
  #pragma omp single
  {
    #pragma omp taskgroup task_reduction(+: a) task_reduction(*: b)
    {
      int i;
      for (i = 1; i <= 5; ++i) {
        #pragma omp task in_reduction(+: a) in_reduction(*: b)
        {
          a += i;
          b *= i;
          #pragma omp task in_reduction(+: a)
          {
            a += i;
          }
        }
      }
    }
  }

  if (a != 30) {
    fprintf(stderr, "error: a != 30. Instead a = %d\n", a);
    exit(EXIT_FAILURE);
  }
  if (b != 120) {
    fprintf(stderr, "error: b != 120. Instead b = %d\n", b);
    exit(EXIT_FAILURE);
  }

  return EXIT_SUCCESS;
}
