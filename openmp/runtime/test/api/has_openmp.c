// RUN: %libomp-compile-and-run
#include <stdio.h>
#include <stdlib.h>
#include "omp_testsuite.h"

int test_has_openmp()
{
  int rvalue = 0;
#ifdef _OPENMP
  rvalue = 1;
#endif
  return (rvalue);
}

int main()
{
  int i;
  int num_failed=0;
  if(!test_has_openmp()) {
    num_failed++;
  }
  return num_failed;
}
