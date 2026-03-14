// RUN: %libomp-compile-and-run
// Linking fails for icc 18
// UNSUPPORTED: icc-18

#include <stdio.h>
#include "omp_testsuite.h"

int test_omp_get_device_num()
{
  /* checks that omp_get_device_num */
  int device_num = omp_get_device_num();

  return (device_num == omp_get_num_devices());
}

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_get_device_num()) {
      num_failed++;
    }
  }
  return num_failed;
}
