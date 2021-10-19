// RUN: %libomp-compile-and-run
#include <stdio.h>
#include <stdlib.h>
#include "omp_testsuite.h"
#include "omp_my_sleep.h"

#define NTIMES 100

// This is the error % threshold. Be generous with the error threshold since
// this test may be run in parallel with many other tests it may throw off the
// sleep timing.
#define THRESHOLD 100.0

double test_omp_get_wtime(double desired_wait_time) {
  double start;
  double end;
  start = 0;
  end = 0;
  start = omp_get_wtime();
  my_sleep(desired_wait_time);
  end = omp_get_wtime();
  return end - start;
}

int compare_times(const void *lhs, const void *rhs) {
  const double *a = (const double *)lhs;
  const double *b = (const double *)rhs;
  return *a - *b;
}

int main() {
  int i, final_count;
  double percent_off;
  double *begin, *end, *ptr;
  double wait_time = 0.01;
  double average = 0.0;
  double n = 0.0;
  double *times = (double *)malloc(sizeof(double) * NTIMES);

  // Get each timing
  for (i = 0; i < NTIMES; i++) {
    times[i] = test_omp_get_wtime(wait_time);
  }

  // Remove approx the "worst" tenth of the timings
  qsort(times, NTIMES, sizeof(double), compare_times);
  begin = times;
  end = times + NTIMES;
  for (i = 0; i < NTIMES / 10; ++i) {
    if (i % 2 == 0)
      begin++;
    else
      end--;
  }

  // Get the average of the remaining timings
  for (ptr = begin, final_count = 0; ptr != end; ++ptr, ++final_count)
    average += times[i];
  average /= (double)final_count;
  free(times);

  // Calculate the percent off of desired wait time
  percent_off = (average - wait_time) / wait_time * 100.0;
  // Should always be positive, but just in case
  if (percent_off < 0)
    percent_off = -percent_off;

  if (percent_off > (double)THRESHOLD) {
    fprintf(stderr, "error: average of %d runs (%lf) is of by %lf%%\n", NTIMES,
            average, percent_off);
    return EXIT_FAILURE;
  }
  printf("pass: average of %d runs (%lf) is only off by %lf%%\n", NTIMES,
         average, percent_off);
  return EXIT_SUCCESS;
}
