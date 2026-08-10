// RUN: %libomp-compile-and-run
// UNSUPPORTED: gcc-4, gcc-5, gcc-6, gcc-7, gcc-8

// support for taskwait with depend clause introduced in clang-14
// UNSUPPORTED: clang-5, clang-6, clang-6, clang-8, clang-9, clang-10, clang-11,
// clang-12, clang-13

// icc does not yet support taskwait with depend clause
// XFAIL: icc

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "omp_my_sleep.h"

int a = 0, b = 0;
int task_grabbed = 0, task_can_proceed = 0;
int task2_grabbed = 0, task2_can_proceed = 0;

static void wait_on_flag(int *flag) {
  int flag_value;
  int timelimit = 30;
  int secs = 0;
  do {
    #pragma omp atomic read
    flag_value = *flag;
    my_sleep(1.0);
    secs++;
    if (secs == timelimit) {
      fprintf(stderr, "error: timeout in wait_on_flag()\n");
      exit(EXIT_FAILURE);
    }
  } while (flag_value == 0);
}

static void signal_flag(int *flag) {
  #pragma omp atomic
  (*flag)++;
}

int main(int argc, char** argv) {

  // Ensure two threads are running
  int num_threads = omp_get_max_threads();
  if (num_threads < 2)
    omp_set_num_threads(2);

  #pragma omp parallel shared(a)
  {
    int a_value;
    // Let us be extra safe here
    if (omp_get_num_threads() > 1) {
      #pragma omp single nowait
      {
        // Schedule independent child task that
        // waits to be flagged after sebsequent taskwait depend()
        #pragma omp task
        {
          signal_flag(&task_grabbed);
          wait_on_flag(&task_can_proceed);
        }
        // Let another worker thread grab the task to execute
        wait_on_flag(&task_grabbed);
        // This should be ignored since the task above has
        // no dependency information
        #pragma omp taskwait depend(inout: a)
        // Signal the independent task to proceed
        signal_flag(&task_can_proceed);

        // Schedule child task with dependencies that taskwait does
        // not care about
        #pragma omp task depend(inout: b)
        {
          signal_flag(&task2_grabbed);
          wait_on_flag(&task2_can_proceed);
          #pragma omp atomic
          b++;
        }
        // Let another worker thread grab the task to execute
        wait_on_flag(&task2_grabbed);
        // This should be ignored since the task above has
        // dependency information on b instead of a
        #pragma omp taskwait depend(inout: a)
        // Signal the task to proceed
        signal_flag(&task2_can_proceed);

        // Generate one child task for taskwait
        #pragma omp task shared(a) depend(inout: a)
        {
          my_sleep(1.0);
          #pragma omp atomic
          a++;
        }
        #pragma omp taskwait depend(inout: a)

        #pragma omp atomic read
        a_value = a;

        if (a_value != 1) {
          fprintf(stderr, "error: dependent task was not executed before "
                          "taskwait finished\n");
          exit(EXIT_FAILURE);
        }
      } // #pragma omp single
    } // if (num_threads > 1)
  } // #pragma omp parallel

  return EXIT_SUCCESS;
}
