// RUN: %libomp-compile && env KMP_ENABLE_TASK_THROTTLING=0 %libomp-run
// RUN: %libomp-compile && env KMP_ENABLE_TASK_THROTTLING=1 %libomp-run

#include<omp.h>
#include<stdlib.h>
#include<string.h>

/**
 * Test the task throttling behavior of the runtime.
 * Unless OMP_NUM_THREADS is 1, the master thread pushes tasks to its own tasks
 * queue until either of the following happens:
 *   - the task queue is full, and it starts serializing tasks
 *   - all tasks have been pushed, and it can begin execution
 * The idea is to create a huge number of tasks which execution are blocked
 * until the master thread comes to execute tasks (they need to be blocking,
 * otherwise the second thread will start emptying the queue).
 * At this point we can check the number of enqueued tasks: iff all tasks have
 * been enqueued, then there was no task throttling.
 * Otherwise there has been some sort of task throttling.
 * If what we detect doesn't match the value of the environment variable, the
 * test is failed.
 */


#define NUM_TASKS 2000


int main()
{
  int i;
  int block = 1;
  int throttling = strcmp(getenv("KMP_ENABLE_TASK_THROTTLING"), "1") == 0;
  int enqueued = 0;
  int failed = -1;

  #pragma omp parallel num_threads(2)
  #pragma omp master
  {
    for (i = 0; i < NUM_TASKS; i++) {
      enqueued++;
      #pragma omp task
      {
        int tid;
        tid = omp_get_thread_num();
        if (tid == 0) {
          // As soon as the master thread starts executing task we should unlock
          // all tasks, and detect the test failure if it has not been done yet.
          if (failed < 0)
            failed = throttling ? enqueued == NUM_TASKS : enqueued < NUM_TASKS;
          block = 0;
        }
        while (block)
          ;
      }
    }
    block = 0;
  }

  return failed;
}
