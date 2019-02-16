// RUN: %libomp-compile-and-run
#include <stdio.h>
#include "omp_testsuite.h"

int test_omp_get_num_devices()
{
  /* checks that omp_get_device_num */
  int num_devices = omp_get_num_devices();

  return (num_devices == 0);
}

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_get_num_devices()) {
      num_failed++;
    }
  }
  return num_failed;
}
