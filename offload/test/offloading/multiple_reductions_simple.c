// RUN: %libomptarget-compile-run-and-check-generic
// RUN: %libomptarget-compileopt-run-and-check-generic

#include <stdio.h>

int main(int argc, char **argv) {

  unsigned s1 = 0, s2 = 1;
#pragma omp target teams distribute parallel for reduction(+ : s1, s2)
  for (int i = 0; i < 10000; ++i) {
    s1 += i;
    s2 += i;
  }

  // CHECK: 49995000 : 49995001
  printf("%i : %i\n", s1, s2);
}
