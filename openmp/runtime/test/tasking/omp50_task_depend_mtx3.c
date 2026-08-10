// RUN: %libomp-compile-and-run
// UNSUPPORTED: gcc-4, gcc-5, gcc-6, gcc-7, gcc-8
// UNSUPPORTED: clang-3, clang-4, clang-5, clang-6, clang-7, clang-8
// TODO: update expected result when icc supports mutexinoutset
// XFAIL: icc

// Tests OMP 5.0 task dependences "mutexinoutset", emulates compiler codegen
// Mutually exclusive tasks get same input dependency info array
//
// Task tree created:
//      task0 task1
//         \    / \
//         task2   task5
//           / \
//       task3  task4
//       /   \
//  task6 <-->task7  (these two are mutually exclusive)
//       \    /
//       task8
//
#include <stdio.h>
#include <omp.h>
#include "omp_my_sleep.h"

static int checker = 0; // to check if two tasks run simultaneously
static int err = 0;
#ifndef DELAY
#define DELAY 0.1
#endif

int mutex_task(int task_id) {
  int th = omp_get_thread_num();
  #pragma omp atomic
    ++checker;
  printf("task %d, th %d\n", task_id, th);
  if (checker != 1) {
    err++;
    printf("Error1, checker %d != 1\n", checker);
  }
  my_sleep(DELAY);
  if (checker != 1) {
    err++;
    printf("Error2, checker %d != 1\n", checker);
  }
  #pragma omp atomic
    --checker;
  return 0;
}

int main()
{
  int i1,i2,i3,i4;
  omp_set_num_threads(2);
  #pragma omp parallel
  {
    #pragma omp single nowait
    {
      int t = omp_get_thread_num();
      #pragma omp task depend(in: i1, i2)
      { int th = omp_get_thread_num();
        printf("task 0_%d, th %d\n", t, th);
        my_sleep(DELAY); }
      #pragma omp task depend(in: i1, i3)
      { int th = omp_get_thread_num();
        printf("task 1_%d, th %d\n", t, th);
        my_sleep(DELAY); }
      #pragma omp task depend(in: i2) depend(out: i1)
      { int th = omp_get_thread_num();
        printf("task 2_%d, th %d\n", t, th);
        my_sleep(DELAY); }
      #pragma omp task depend(in: i1)
      { int th = omp_get_thread_num();
        printf("task 3_%d, th %d\n", t, th);
        my_sleep(DELAY); }
      #pragma omp task depend(out: i2)
      { int th = omp_get_thread_num();
        printf("task 4_%d, th %d\n", t, th);
        my_sleep(DELAY+0.1); } // wait a bit longer than task 3
      #pragma omp task depend(out: i3)
      { int th = omp_get_thread_num();
        printf("task 5_%d, th %d\n", t, th);
        my_sleep(DELAY); }

      #pragma omp task depend(mutexinoutset: i1, i4)
      { mutex_task(6); }
      #pragma omp task depend(mutexinoutset: i1, i4)
      { mutex_task(7); }

      #pragma omp task depend(in: i1)
      { int th = omp_get_thread_num();
        printf("task 8_%d, th %d\n", t, th);
        my_sleep(DELAY); }
    } // single
  } // parallel
  if (err == 0) {
    printf("passed\n");
    return 0;
  } else {
    printf("failed\n");
    return 1;
  }
}
