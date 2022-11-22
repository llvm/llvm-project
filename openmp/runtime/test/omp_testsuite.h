/* Global headerfile of the OpenMP Testsuite */

#ifndef OMP_TESTSUITE_H
#define OMP_TESTSUITE_H

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/* General                                                */
/**********************************************************/
#define LOOPCOUNT 1000 /* Number of iterations to slit amongst threads */
#define REPETITIONS 10 /* Number of times to run each test */

/* following times are in seconds */
#define SLEEPTIME 1

/* Definitions for tasks                                  */
/**********************************************************/
#define NUM_TASKS 25
#define MAX_TASKS_PER_THREAD 5

// Functions that call a parallel region that does very minimal work
// Some compilers may optimize away an empty parallel region
volatile int g_counter__;

// If nthreads == 0, then do not use num_threads() clause
static void go_parallel() {
  g_counter__ = 0;
  #pragma omp parallel
  {
    #pragma omp atomic
    g_counter__++;
  }
}

static void go_parallel_nthreads(int nthreads) {
  g_counter__ = 0;
  #pragma omp parallel num_threads(nthreads)
  {
    #pragma omp atomic
    g_counter__++;
  }
}

static void go_parallel_spread() {
  g_counter__ = 0;
  #pragma omp parallel proc_bind(spread)
  {
    #pragma omp atomic
    g_counter__++;
  }
}

static void go_parallel_close() {
  g_counter__ = 0;
  #pragma omp parallel proc_bind(close)
  {
    #pragma omp atomic
    g_counter__++;
  }
}

static void go_parallel_master() {
  g_counter__ = 0;
  #pragma omp parallel proc_bind(master)
  {
    #pragma omp atomic
    g_counter__++;
  }
}

static inline int get_exit_value() {
  return ((g_counter__ == -1) ? EXIT_FAILURE : EXIT_SUCCESS);
}

#ifdef  _WIN32
// Windows versions of pthread_create() and pthread_join()
# include <windows.h>
typedef HANDLE pthread_t;

// encapsulates the information about a pthread-callable function
struct thread_func_info_t {
  void* (*start_routine)(void*);
  void* arg;
};

// call the void* start_routine(void*);
static DWORD WINAPI __thread_func_wrapper(LPVOID lpParameter) {
  struct thread_func_info_t* function_information;
  function_information = (struct thread_func_info_t*)lpParameter;
  function_information->start_routine(function_information->arg);
  free(function_information);
  return 0;
}

// attr is ignored
static int pthread_create(pthread_t *thread, void *attr,
                          void *(*start_routine) (void *), void *arg) {
  pthread_t pthread;
  struct thread_func_info_t* info;
  info = (struct thread_func_info_t*)malloc(sizeof(struct thread_func_info_t));
  info->start_routine = start_routine;
  info->arg = arg;
  pthread = CreateThread(NULL, 0, __thread_func_wrapper, info, 0, NULL);
  if (pthread == NULL) {
    fprintf(stderr, "CreateThread() failed: Error #%u.\n", GetLastError());
    exit(1);
  }
  *thread = pthread;
  return 0;
}
// retval is ignored for now
static int pthread_join(pthread_t thread, void **retval) {
  int rc;
  rc = WaitForSingleObject(thread, INFINITE);
  if (rc == WAIT_FAILED) {
    fprintf(stderr, "WaitForSingleObject() failed: Error #%u.\n",
            GetLastError());
    exit(1);
  }
  rc = CloseHandle(thread);
  if (rc == 0) {
    fprintf(stderr, "CloseHandle() failed: Error #%u.\n", GetLastError());
    exit(1);
  }
  return 0;
}
#else
# include <pthread.h>
#endif

#endif
