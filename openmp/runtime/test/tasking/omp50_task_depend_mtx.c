// RUN: %libomp-compile-and-run

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

#ifdef _WIN32
#include <windows.h>
#define mysleep(n) Sleep(n)
#else
#include <unistd.h>
#define mysleep(n) usleep((n)*1000)
#endif

static int checker = 0; // to check if two tasks run simultaneously
static int err = 0;
#ifndef DELAY
#define DELAY 100
#endif

// ---------------------------------------------------------------------------
// internal data to emulate compiler codegen
typedef int(*entry_t)(int, int**);
typedef struct DEP {
  size_t addr;
  size_t len;
  int flags;
} dep;
typedef struct ID {
  int reserved_1;
  int flags;
  int reserved_2;
  int reserved_3;
  char *psource;
} id;

int thunk(int gtid, int** pshareds) {
  int t = **pshareds;
  int th = omp_get_thread_num();
  #pragma omp atomic
    ++checker;
  printf("task __%d, th %d\n", t, th);
  if (checker != 1) {
    err++;
    printf("Error1, checker %d != 1\n", checker);
  }
  mysleep(DELAY);
  if (checker != 1) {
    err++;
    printf("Error2, checker %d != 1\n", checker);
  }
  #pragma omp atomic
    --checker;
  return 0;
}

#ifdef __cplusplus
extern "C" {
#endif
int __kmpc_global_thread_num(id*);
extern int** __kmpc_omp_task_alloc(id *loc, int gtid, int flags,
                                   size_t sz, size_t shar, entry_t rtn);
int
__kmpc_omp_task_with_deps(id *loc, int gtid, int **task, int nd, dep *dep_lst,
                          int nd_noalias, dep *noalias_dep_lst);
static id loc = {0, 2, 0, 0, ";file;func;0;0;;"};
#ifdef __cplusplus
} // extern "C"
#endif
// End of internal data
// ---------------------------------------------------------------------------

int main()
{
  int i1,i2,i3,i4;
  omp_set_num_threads(2);
  #pragma omp parallel
  {
    #pragma omp single nowait
    {
      dep sdep[2];
      int **ptr;
      int gtid = __kmpc_global_thread_num(&loc);
      int t = omp_get_thread_num();
      #pragma omp task depend(in: i1, i2)
      { int th = omp_get_thread_num();
        printf("task 0_%d, th %d\n", t, th);
        mysleep(DELAY); }
      #pragma omp task depend(in: i1, i3)
      { int th = omp_get_thread_num();
        printf("task 1_%d, th %d\n", t, th);
        mysleep(DELAY); }
      #pragma omp task depend(in: i2) depend(out: i1)
      { int th = omp_get_thread_num();
        printf("task 2_%d, th %d\n", t, th);
        mysleep(DELAY); }
      #pragma omp task depend(in: i1)
      { int th = omp_get_thread_num();
        printf("task 3_%d, th %d\n", t, th);
        mysleep(DELAY); }
      #pragma omp task depend(out: i2)
      { int th = omp_get_thread_num();
        printf("task 4_%d, th %d\n", t, th);
        mysleep(DELAY+5); } // wait a bit longer than task 3
      #pragma omp task depend(out: i3)
      { int th = omp_get_thread_num();
        printf("task 5_%d, th %d\n", t, th);
        mysleep(DELAY); }
// compiler codegen start
      // task1
      ptr = __kmpc_omp_task_alloc(&loc, gtid, 0, 28, 16, thunk);
      sdep[0].addr = (size_t)&i1;
      sdep[0].len = 0;   // not used
      sdep[0].flags = 4; // mx
      sdep[1].addr = (size_t)&i4;
      sdep[1].len = 0;   // not used
      sdep[1].flags = 4; // mx
      **ptr = t + 10; // init single shared variable
      __kmpc_omp_task_with_deps(&loc, gtid, ptr, 2, sdep, 0, 0);

      // task2
      ptr = __kmpc_omp_task_alloc(&loc, gtid, 0, 28, 16, thunk);
      **ptr = t + 20; // init single shared variable
      __kmpc_omp_task_with_deps(&loc, gtid, ptr, 2, sdep, 0, 0);
// compiler codegen end
      #pragma omp task depend(in: i1)
      { int th = omp_get_thread_num();
        printf("task 8_%d, th %d\n", t, th);
        mysleep(DELAY); }
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
