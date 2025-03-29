// RUN: %libomptarget-compile-run-and-check-generic
// RUN: %libomptarget-compileopt-run-and-check-generic

#include <omp.h>
#include <stdio.h>

__attribute__((optnone)) void optnone() {}

int main() {
  int i = 0;
#pragma omp target teams num_teams(1) map(tofrom : i)
  {
    optnone();
#pragma omp parallel
    if (omp_get_thread_num() == 0)
      ++i;
#pragma omp parallel
    if (omp_get_thread_num() == 0)
      ++i;
#pragma omp parallel
    if (omp_get_thread_num() == 0)
      ++i;
#pragma omp parallel
    if (omp_get_thread_num() == 0)
      ++i;
  }
  // CHECK: 4
  printf("%i\n", i);
}
