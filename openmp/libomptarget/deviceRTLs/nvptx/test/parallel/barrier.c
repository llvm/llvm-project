// RUN: %compile-run-and-check

#include <omp.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  int data, out, flag = 0;
#pragma omp target teams num_teams(2) map(tofrom                               \
                                          : out) map(to                        \
                                                     : data, flag)             \
    thread_limit(1)
#pragma omp parallel num_threads(1)
  {
    if (omp_get_team_num() == 0) {
      /* Write to the data buffer that will be read by thread in team 1 */
      data = 42;
/* Flush data to thread in team 1 */
#pragma omp barrier
      /* Set flag to release thread in team 1 */
#pragma omp atomic write
      flag = 1;
    } else if (omp_get_team_num() == 1) {
      /* Loop until we see the update to the flag */
      int val;
      do {
#pragma omp atomic read
        val = flag;
      } while (val < 1);
      out = data;
#pragma omp barrier
    }
  }
  // CHECK: out=42.
  /* Value of out will be 42 */
  printf("out=%d.\n", out);
  return !(out == 42);
}
