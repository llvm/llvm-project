// RUN: %libomptarget-compilexx-run-and-check-generic
// RUN: %libomptarget-compileoptxx-run-and-check-generic

#include <omp.h>
#include <stdio.h>
__attribute__((optnone)) void optnone(void) {}

int main() {
  int sum = 0, nt;
#pragma omp target teams map(tofrom : sum, nt) num_teams(1)
  {
    nt = 3 * omp_get_max_threads();
    optnone();
#pragma omp parallel reduction(+ : sum)
    sum += 1;
#pragma omp parallel reduction(+ : sum)
    sum += 1;
#pragma omp parallel reduction(+ : sum)
    sum += 1;
  }
  // CHECK: nt: [[NT:.*]]
  // CHECK: sum: [[NT]]
  printf("nt: %i\n", nt);
  printf("sum: %i\n", sum);
}
