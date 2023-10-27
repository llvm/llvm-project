// RUN: %libomp-compile && env OMP_MAX_TASK_PRIORITY=42 %libomp-run

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int a = 0;

int main(void) {
  int i;
  int max_task_priority = omp_get_max_task_priority();
  if (max_task_priority != 42) {
    fprintf(stderr,
            "error: omp_get_max_task_priority() returned %d instead of 42\n",
            max_task_priority);
    exit(EXIT_FAILURE);
  }

  for (i = 0; i < 250; ++i) {
    #pragma omp parallel
    {
      #pragma omp task priority(42)
      {
        #pragma omp atomic
        a++;
      }
    }
  }

  printf("a = %d\n", a);

  return EXIT_SUCCESS;
}
