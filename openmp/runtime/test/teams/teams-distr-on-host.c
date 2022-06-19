// The test supposes no offload, pure host execution.
// It checks that the bug in implementation of distribute construct is fixed.

// RUN: %libomp-compile-and-run
// UNSUPPORTED: icc

#include <stdio.h>
#include <omp.h>

int main()
{
  const int size = 4;
  int wrong_counts = 0;
  omp_set_num_threads(2);
  #pragma omp parallel reduction(+:wrong_counts)
  {
    int i;
    int A[size];
    int th = omp_get_thread_num();
    for(i = 0; i < size; i++)
      A[i] = 0;

    #pragma omp target teams distribute map(tofrom: A[:size]) private(i)
    for(i = 0; i < size; i++)
    {
      A[i] = i;
      printf("th %d, team %d, i %d\n", th, omp_get_team_num(), i);
    }
    #pragma omp critical
    {
      printf("tid = %d\n", th);
      for(i = 0; i < size; i++)
      {
        if (A[i] != i) wrong_counts++;
        printf("  %d", A[i]);
      }
      printf("\n");
    }
  }
  if (wrong_counts) {
    printf("failed\n");
  } else {
    printf("passed\n");
  }
  return wrong_counts;
}
