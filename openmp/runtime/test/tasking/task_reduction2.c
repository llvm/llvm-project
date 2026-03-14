// RUN: %libomp-compile-and-run
//
// XFAIL: icc
// UNSUPPORTED: clang-4, clang-5, clang-6, clang-7, clang-8, clang-9, clang-10
// UNSUPPORTED: gcc-4, gcc-5, gcc-6, gcc-7, gcc-8

#include <stdio.h>
#include <stdlib.h>

int a = 0, b = 1;

int main(int argc, char **argv) {

  #pragma omp parallel
  {
    int i;
    #pragma omp for reduction(task, +: a) reduction(task, *: b)
    for (i = 1; i <= 5; ++i) {
      #pragma omp task in_reduction(+: a) in_reduction(*: b)
      {
        a += i;
        b *= i;
      }
    }
  }

  if (a != 15) {
    fprintf(stderr, "error: a != 15. Instead a = %d\n", a);
    exit(EXIT_FAILURE);
  }
  if (b != 120) {
    fprintf(stderr, "error: b != 120. Instead b = %d\n", b);
    exit(EXIT_FAILURE);
  }

  return EXIT_SUCCESS;
}
