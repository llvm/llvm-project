// RUN: %libomp-compile -lpthread && %libomp-run
#include <stdio.h>
#include "omp_testsuite.h"

#define NUM_THREADS 10

/*
 After hot teams were enabled by default, the library started using levels
 kept in the team structure.  The levels are broken in case foreign thread
 exits and puts its team into the pool which is then re-used by another foreign
 thread. The broken behavior observed is when printing the levels for each
 new team, one gets 1, 2, 1, 2, 1, 2, etc.  This makes the library believe that
 every other team is nested which is incorrect.  What is wanted is for the
 levels to be 1, 1, 1, etc.
*/

int a = 0;
int level;

typedef struct thread_arg_t {
  int iterations;
} thread_arg_t;

void* thread_function(void* arg) {
  int i;
  thread_arg_t* targ = (thread_arg_t*)arg;
  int iterations = targ->iterations;
  #pragma omp parallel private(i)
  {
    // level should always be 1
    #pragma omp single
    level = omp_get_level();

    #pragma omp for
    for(i = 0; i < iterations; i++) {
      #pragma omp atomic
      a++;
    }
  }
}

int test_omp_team_reuse()
{
  int i;
  int success = 1;
  pthread_t thread[NUM_THREADS];
  thread_arg_t thread_arg[NUM_THREADS];
  // launch NUM_THREADS threads, one at a time to perform thread_function()
  for(i = 0; i < NUM_THREADS; i++) {
    thread_arg[i].iterations = i + 1;
    pthread_create(thread+i, NULL, thread_function, thread_arg+i);
    pthread_join(*(thread+i), NULL);
    // level read in thread_function()'s parallel region should be 1
    if(level != 1) {
      fprintf(stderr, "error: for pthread %d level should be 1 but "
                      "instead equals %d\n", i, level);
      success = 0;
    }
  }
  // make sure the for loop works
  int known_sum = (NUM_THREADS * (NUM_THREADS+1)) / 2;
  if(a != known_sum) {
    fprintf(stderr, "a should be %d but instead equals %d\n", known_sum, a);
    success = 0;
  }
  return success;
}

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    a = 0;
    if(!test_omp_team_reuse()) {
      num_failed++;
    }
  }
  return num_failed;
}
