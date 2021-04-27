// RUN: %libomptarget-compile-run-and-check-generic

#include <stdio.h>
#include <omp.h>

int test_omp_get_num_devices()
{
  /* checks that omp_get_num_devices() > 0 */
  int num_devices = omp_get_num_devices();
  printf("num_devices = %d\n", num_devices);

  #pragma omp target
  {}

  return (num_devices > 0);
}

int main()
{
  int i;
  int failed=0;

  if (!test_omp_get_num_devices()) {
    failed++;
  }
  if (failed)
    printf("FAIL\n");
  else
    printf("PASS\n");
  return failed;
}

// CHECK: PASS
