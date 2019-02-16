// RUN: %libomp-compile-and-run
#include <stdio.h>
#include "omp_testsuite.h"

int test_omp_get_wtick()
{
  double tick;
  tick = -1.;
  tick = omp_get_wtick ();
  return ((tick > 0.0) && (tick <= 0.01));
}

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_get_wtick()) {
      num_failed++;
    }
  }
  return num_failed;
}
