// RUN: %libomp-compile && env OMP_MAX_TASK_PRIORITY='2' %libomp-run

// Test OMP 4.5 task priorities
// Higher priority task supposed to be executed before lower priority task.

#include <stdio.h>
#include <omp.h>

#include "omp_my_sleep.h"
// delay(n) - sleep n ms
#define delay(n) my_sleep(((double)n)/1000.0)

int main ( void ) {
  int passed;
  passed = (omp_get_max_task_priority() == 2);
  printf("Got %d max priority via env\n", omp_get_max_task_priority());
  if(!passed) {
    printf( "failed\n" );
    return 1;
  }
  printf("parallel 1 spawns 4 tasks for primary thread to execute\n");
  #pragma omp parallel num_threads(2)
  {
    int th = omp_get_thread_num();
    if (th == 0) // primary thread
    {
      #pragma omp task priority(1)
      { // middle priority
        int val, t = omp_get_thread_num();
        #pragma omp atomic capture
          val = passed++;
        printf("P1:    val = %d, thread gen %d, thread exe %d\n", val, th, t);
        delay(10); // sleep 10 ms
      }
      #pragma omp task priority(2)
      { // high priority
        int val, t = omp_get_thread_num();
        #pragma omp atomic capture
          val = passed++;
        printf("P2:    val = %d, thread gen %d, thread exe %d\n", val, th, t);
        delay(20); // sleep 20 ms
      }
      #pragma omp task priority(0)
      { // low priority specified explicitly
        int val, t = omp_get_thread_num();
        #pragma omp atomic capture
          val = passed++;
        printf("P0exp: val = %d, thread gen %d, thread exe %d\n", val, th, t);
        delay(1); // sleep 1 ms
      }
      #pragma omp task
      { // low priority by default
        int val, t = omp_get_thread_num();
        #pragma omp atomic capture
          val = passed++;
        printf("P0imp: val = %d, thread gen %d, thread exe %d\n", val, th, t);
        delay(1); // sleep 1 ms
      }
    } else {
      // wait for the primary thread to finish all tasks
      int wait = 0;
      do {
        delay(5);
        #pragma omp atomic read
          wait = passed;
      } while (wait < 5);
    }
  }
  printf("parallel 2 spawns 4 tasks for worker thread to execute\n");
  #pragma omp parallel num_threads(2)
  {
    int th = omp_get_thread_num();
    if (th == 0) // primary thread
    {
      #pragma omp task priority(1)
      { // middle priority
        int val, t = omp_get_thread_num();
        #pragma omp atomic capture
          val = passed++;
        printf("P1:    val = %d, thread gen %d, thread exe %d\n", val, th, t);
        delay(10); // sleep 10 ms
      }
      #pragma omp task priority(2)
      { // high priority
        int val, t = omp_get_thread_num();
        #pragma omp atomic capture
          val = passed++;
        printf("P2:    val = %d, thread gen %d, thread exe %d\n", val, th, t);
        delay(20); // sleep 20 ms
      }
      #pragma omp task priority(0)
      { // low priority specified explicitly
        int val, t = omp_get_thread_num();
        #pragma omp atomic capture
          val = passed++;
        printf("P0exp: val = %d, thread gen %d, thread exe %d\n", val, th, t);
        delay(1); // sleep 1 ms
      }
      #pragma omp task
      { // low priority by default
        int val, t = omp_get_thread_num();
        #pragma omp atomic capture
          val = passed++;
        printf("P0imp: val = %d, thread gen %d, thread exe %d\n", val, th, t);
        delay(1); // sleep 1 ms
      }
      // signal creation of all tasks: passed = 5 + 1 = 6
      #pragma omp atomic
        passed++;
      // wait for completion of all 4 tasks
      int wait = 0;
      do {
        delay(5);
        #pragma omp atomic read
          wait = passed;
      } while (wait < 10); // passed = 6 + 4 = 10
    } else {
      // wait for the primary thread to create all tasks
      int wait = 0;
      do {
        delay(5);
        #pragma omp atomic read
          wait = passed;
      } while (wait < 6);
      // go execute 4 tasks created by primary thread
    }
  }
  if (passed != 10) {
    printf("failed, passed = %d (should be 10)\n", passed);
    return 1;
  }
  printf("passed\n");
  return 0;
}
// CHECK: parallel 1
// CHECK-NEXT: P2
// CHECK-NEXT: P1
// CHECK-NEXT: P0
// CHECK-NEXT: P0
// CHECK-NEXT: parallel 2
// CHECK-NEXT: P2
// CHECK-NEXT: P1
// CHECK-NEXT: P0
// CHECK-NEXT: P0
// CHECK: passed
