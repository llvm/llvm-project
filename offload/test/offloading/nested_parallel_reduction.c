// Both parallel regions have to be nested for this to reproduce, and only at
// -O2 and above; either one alone is correct at every optimization level.

// RUN: %libomptarget-compileopt-generic
// RUN: %libomptarget-run-generic | %fcheck-generic

// REQUIRES: gpu

#include <stdio.h>

#define N 5

int main(void) {
  long aa = 0;
  int ng = 6, cmom = 4, nxyz = 5;

#pragma omp target teams distribute num_teams(nxyz) thread_limit(4)            \
    map(tofrom : aa)
  for (int gid = 0; gid < nxyz; gid++) {
#pragma omp parallel for collapse(2)
    for (unsigned g = 0; g < ng; g++)
      for (unsigned l = 0; l < cmom - 1; l++) {
        int a = 0;
        for (int ii = 0; ii < N + 2; ii++) {
#pragma omp parallel for reduction(+ : a)
          for (int i = 0; i < N; i++)
            a += i;
        }
#pragma omp atomic
        aa += a;
      }
  }

  long expected = (long)ng * (cmom - 1) * nxyz * (N * (N - 1) / 2) * (N + 2);
  printf("aa = %ld, expected %ld\n", aa, expected);
  return aa != expected;
}

// CHECK: aa = 6300, expected 6300
