// RUN: %libomp-compile-and-run
#include <stdio.h>
#include <stdlib.h>

int a;

void run(int nteams, int nth) {
  a = 0;
#pragma omp teams num_teams(nteams)
  {
#pragma omp parallel num_threads(nth)
    {
#pragma omp task
      {
#pragma omp atomic
        a++;
      }
    }
  }
  if (a == 0)
    exit(EXIT_FAILURE);
}

int main() {
  int i, nteams, nth;
  for (nteams = 1; nteams <= 2; ++nteams)
    for (nth = 1; nth <= 3; ++nth)
      for (i = 0; i < 10; ++i) {
        printf("run(%d, %d)\n", nteams, nth);
        run(nteams, nth);
      }
  return EXIT_SUCCESS;
}
