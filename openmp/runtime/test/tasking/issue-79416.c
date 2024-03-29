// RUN: %libomp-compile-and-run
#include <stdio.h>

void run(int teams, int th) {
#pragma omp teams num_teams(teams)
#pragma omp parallel num_threads(th)
#pragma omp task
  {}
}

int main() {
  int i;
  for (i = 0; i < 10; ++i) {
    printf("run(1, 1)\n");
    run(1, 1);
    printf("run(1, 2)\n");
    run(1, 2);
    printf("run(1, 3)\n");
    run(1, 3);
    printf("run(2, 1)\n");
    run(2, 1);
    printf("run(2, 2)\n");
    run(2, 2);
    printf("run(2, 3)\n");
    run(2, 3);
  }
  return 0;
}
