// RUN: %libomp-compile
// RUN: env OMP_SCHEDULE=static %libomp-run 1 0
// RUN: env OMP_SCHEDULE=static,10 %libomp-run 1 10
// RUN: env OMP_SCHEDULE=dynamic %libomp-run 2 1
// RUN: env OMP_SCHEDULE=dynamic,11 %libomp-run 2 11
// RUN: env OMP_SCHEDULE=guided %libomp-run 3 1
// RUN: env OMP_SCHEDULE=guided,12 %libomp-run 3 12
// RUN: env OMP_SCHEDULE=auto %libomp-run 4 1
// RUN: env OMP_SCHEDULE=trapezoidal %libomp-run 101 1
// RUN: env OMP_SCHEDULE=trapezoidal,13 %libomp-run 101 13
// RUN: env OMP_SCHEDULE=static_steal %libomp-run 102 1
// RUN: env OMP_SCHEDULE=static_steal,14 %libomp-run 102 14

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "omp_testsuite.h"

int sum;
char* correct_kind_string;
omp_sched_t correct_kind;
int correct_chunk_size;

int test_omp_for_runtime()
{
  int sum;
  int known_sum;
  int chunk_size;
  int error;
  omp_sched_t kind;

  sum = 0;
  error = 0;
  known_sum = (LOOPCOUNT * (LOOPCOUNT + 1)) / 2;
  omp_get_schedule(&kind, &chunk_size);

  printf("omp_get_schedule() returns: Schedule = %d, Chunk Size = %d\n",
         kind, chunk_size);
  if (kind != correct_kind) {
    printf("kind(%d) != correct_kind(%d)\n", kind, correct_kind);
    error = 1;
  }
  if (chunk_size != correct_chunk_size) {
    printf("chunk_size(%d) != correct_chunk_size(%d)\n", chunk_size,
           correct_chunk_size);
    error = 1;
  }

  #pragma omp parallel
  {
    int i;
    #pragma omp for schedule(runtime)
    for (i = 1; i <= LOOPCOUNT; i++) {
        #pragma omp critical
        sum+=i;
    }
  }
  if (known_sum != sum) {
    printf("Known Sum = %d, Calculated Sum = %d\n", known_sum, sum);
    error = 1;
  }
  return !error;
}

int main(int argc, char** argv)
{
  int i;
  int num_failed=0;
  if (argc != 3) {
    fprintf(stderr, "usage: %s schedule_kind chunk_size\n", argv[0]);
    fprintf(stderr, "  Run with envirable OMP_SCHEDULE=kind[,chunk_size]\n");
    return 1;
  }
  correct_kind = atoi(argv[1]);
  correct_chunk_size = atoi(argv[2]);

  for (i = 0; i < REPETITIONS; i++) {
    if (!test_omp_for_runtime()) {
      num_failed++;
    }
  }
  return num_failed;
}
