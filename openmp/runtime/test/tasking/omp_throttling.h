/**
 *  This test ensures that a task throttling works
 *
 *  It creates 2 threads (1 producer, 1 consummer)
 *  The producer infinitely create tasks 'T_i' until one executed
 *  The consumer is blocked until any 'T_i' executed
 *  Executing any 'T_i' unblocks the consumer and stop the producer
 *
 *  The assertion tests ensures that the producer does not create more than the
 *  total number of tasks provided by the programmer
 */

#include <assert.h>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>

int main(void) {

  /* check that throttling is enabled (enabled by default) */
  char *throttling_str = getenv("KMP_ENABLE_TASK_THROTTLING");
  assert(throttling_str);
  assert(*throttling_str == '1');

  /* retrieve the maximum number of task for this specific throttling parameter
   */
  char *max_tasks_str = getenv(MAX_ENV_VAR);
  assert(max_tasks_str);
  int max_tasks = atoi(max_tasks_str);
  if (max_tasks <= 0)
    max_tasks = 1;

  /* check that specific throttling under-test is enabled */
  throttling_str = getenv(ENABLE_ENV_VAR);
  assert(throttling_str);
  assert(*throttling_str == '1');

  volatile int done = 0;
  int ntasks = 0;

#pragma omp parallel num_threads(2) default(none)                              \
    shared(max_tasks, done, ntasks)
  {
    if (omp_get_thread_num() == 1)
      while (!done)
        ;

#pragma omp single
    {
      while (!done) {
#if USE_DEPS
#pragma omp task default(none) shared(done) depend(out : max_tasks)
#else
#pragma omp task default(none) shared(done)
#endif
        { done = 1; }

        assert(++ntasks <= max_tasks + 1);
      }
    }
  }
  assert(ntasks == max_tasks + 1);

  return 0;
}
