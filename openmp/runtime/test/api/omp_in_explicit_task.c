// RUN: %libomp-compile-and-run

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main()
{
  int res;
  res = omp_in_explicit_task();
  if (res) {
    printf("error: omp_in_explicit_task: serial1 returned %d\n", res);
    return 1;
  }
  #pragma omp parallel num_threads(2)
  {
    int r = omp_in_explicit_task();
    if (r) {
      printf("error: omp_in_explicit_task: par #%d returned %d\n",
             omp_get_thread_num(), r);
      exit(1);
    }
    #pragma omp task
    {
      int r = omp_in_explicit_task();
      if (!r) {
        printf("error: omp_in_explicit_task: task1 #%d returned %d\n",
               omp_get_thread_num(), r);
        exit(1);
      }
    }
    #pragma omp task
    {
      int r = omp_in_explicit_task();
      if (!r) {
        printf("error: omp_in_explicit_task: task2 #%d returned %d\n",
               omp_get_thread_num(), r);
        exit(1);
      }
    }
  }
  res = omp_in_explicit_task();
  if (res) {
    printf("error: omp_in_explicit_task: serial2 returned %d\n", res);
    return 1;
  }
  printf("passed\n");
  return 0;
}
