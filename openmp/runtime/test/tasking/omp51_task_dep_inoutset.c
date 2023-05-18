// RUN: %libomp-compile-and-run
// RUN: %libomp-cxx-compile-and-run
// UNSUPPORTED: gcc

// Tests OMP 5.0 task dependences "mutexinoutset" and 5.1 "inoutset",
// emulates compiler codegen for new dep kinds
// Mutually exclusive tasks get same input dependency info array
//
// Task tree created:
//      task0 - task1 (in)
//             \
//        task2 - task3 (inoutset)
//             /
//      task3 - task4 (in)
//           /
//  task6 <-->task7  (mutexinoutset)
//       \    /
//       task8 (in)
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

// to check the # of concurrent tasks (must be 1 for MTX, <3 for other kinds)
static int volatile checker = 0;
static int err = 0;
#ifndef DELAY
#define DELAY 100
#endif

// ---------------------------------------------------------------------------
// internal data to emulate compiler codegen
typedef struct DEP {
  size_t addr;
  size_t len;
  unsigned char flags;
} dep;
typedef struct task {
  void** shareds;
  void* entry;
  int part_id;
  void* destr_thunk;
  int priority;
  long long device_id;
  int f_priv;
} task_t;
#define TIED 1
typedef int(*entry_t)(int, task_t*);
typedef struct ID {
  int reserved_1;
  int flags;
  int reserved_2;
  int reserved_3;
  char *psource;
} id;
// thunk routine for tasks with MTX dependency
int thunk_m(int gtid, task_t* ptask) {
  int th = omp_get_thread_num();
  #pragma omp atomic
    ++checker;
  printf("task _%d, th %d\n", ptask->f_priv, th);
  if (checker != 1) { // no more than 1 task at a time
    err++;
    printf("Error1, checker %d != 1\n", checker);
  }
  mysleep(DELAY);
  if (checker != 1) { // no more than 1 task at a time
    err++;
    printf("Error2, checker %d != 1\n", checker);
  }
  #pragma omp atomic
    --checker;
  return 0;
}
// thunk routine for tasks with inoutset dependency
int thunk_s(int gtid, task_t* ptask) {
  int th = omp_get_thread_num();
  #pragma omp atomic
    ++checker;
  printf("task _%d, th %d\n", ptask->f_priv, th);
  if (checker > 2) { // no more than 2 tasks concurrently
    err++;
    printf("Error1, checker %d > 2\n", checker);
  }
  mysleep(DELAY);
  if (checker > 2) { // no more than 2 tasks concurrently
    err++;
    printf("Error2, checker %d > 2\n", checker);
  }
  #pragma omp atomic
    --checker;
  return 0;
}

#ifdef __cplusplus
extern "C" {
#endif
int __kmpc_global_thread_num(id*);
extern task_t* __kmpc_omp_task_alloc(id *loc, int gtid, int flags,
                                     size_t sz, size_t shar, entry_t rtn);
int
__kmpc_omp_task_with_deps(id *loc, int gtid, task_t *task, int nd, dep *dep_lst,
                          int nd_noalias, dep *noalias_dep_lst);
static id loc = {0, 2, 0, 0, ";file;func;0;0;;"};
#ifdef __cplusplus
} // extern "C"
#endif
// End of internal data
// ---------------------------------------------------------------------------

int main()
{
  int i1,i2,i3;
  omp_set_num_threads(4);
  omp_set_dynamic(0);
  #pragma omp parallel
  {
    #pragma omp single nowait
    {
      dep sdep[2];
      task_t *ptr;
      int gtid = __kmpc_global_thread_num(&loc);
      int t = omp_get_thread_num();
      #pragma omp task depend(in: i1, i2)
      { int th = omp_get_thread_num();
        printf("task 0_%d, th %d\n", t, th);
        #pragma omp atomic
          ++checker;
        if (checker > 2) { // no more than 2 tasks concurrently
          err++;
          printf("Error1, checker %d > 2\n", checker);
        }
        mysleep(DELAY);
        if (checker > 2) { // no more than 2 tasks concurrently
          err++;
          printf("Error1, checker %d > 2\n", checker);
        }
        #pragma omp atomic
          --checker;
      }
      #pragma omp task depend(in: i1, i2)
      { int th = omp_get_thread_num();
        printf("task 1_%d, th %d\n", t, th);
        #pragma omp atomic
          ++checker;
        if (checker > 2) { // no more than 2 tasks concurrently
          err++;
          printf("Error1, checker %d > 2\n", checker);
        }
        mysleep(DELAY);
        if (checker > 2) { // no more than 2 tasks concurrently
          err++;
          printf("Error1, checker %d > 2\n", checker);
        }
        #pragma omp atomic
          --checker;
      }
// compiler codegen start
      // task2
      ptr = __kmpc_omp_task_alloc(&loc, gtid, TIED, sizeof(task_t), 0, thunk_s);
      sdep[0].addr = (size_t)&i1;
      sdep[0].len = 0;   // not used
      sdep[0].flags = 1; // IN
      sdep[1].addr = (size_t)&i2;
      sdep[1].len = 0;   // not used
      sdep[1].flags = 8; // INOUTSET
      ptr->f_priv = t + 10; // init single first-private variable
      __kmpc_omp_task_with_deps(&loc, gtid, ptr, 2, sdep, 0, 0);

      // task3
      ptr = __kmpc_omp_task_alloc(&loc, gtid, TIED, sizeof(task_t), 0, thunk_s);
      ptr->f_priv = t + 20; // init single first-private variable
      __kmpc_omp_task_with_deps(&loc, gtid, ptr, 2, sdep, 0, 0);
// compiler codegen end
      t = omp_get_thread_num();
      #pragma omp task depend(in: i1, i2)
      { int th = omp_get_thread_num();
        printf("task 4_%d, th %d\n", t, th);
        #pragma omp atomic
          ++checker;
        if (checker > 2) { // no more than 2 tasks concurrently
          err++;
          printf("Error1, checker %d > 2\n", checker);
        }
        mysleep(DELAY);
        if (checker > 2) { // no more than 2 tasks concurrently
          err++;
          printf("Error1, checker %d > 2\n", checker);
        }
        #pragma omp atomic
          --checker;
      }
      #pragma omp task depend(in: i1, i2)
      { int th = omp_get_thread_num();
        printf("task 5_%d, th %d\n", t, th);
        #pragma omp atomic
          ++checker;
        if (checker > 2) { // no more than 2 tasks concurrently
          err++;
          printf("Error1, checker %d > 2\n", checker);
        }
        mysleep(DELAY);
        if (checker > 2) { // no more than 2 tasks concurrently
          err++;
          printf("Error1, checker %d > 2\n", checker);
        }
        #pragma omp atomic
          --checker;
      }
// compiler codegen start
      // task6
      ptr = __kmpc_omp_task_alloc(&loc, gtid, TIED, sizeof(task_t), 0, thunk_m);
      sdep[0].addr = (size_t)&i1;
      sdep[0].len = 0;   // not used
      sdep[0].flags = 4; // MUTEXINOUTSET
      sdep[1].addr = (size_t)&i3;
      sdep[1].len = 0;   // not used
      sdep[1].flags = 4; // MUTEXINOUTSET
      ptr->f_priv = t + 30; // init single first-private variable
      __kmpc_omp_task_with_deps(&loc, gtid, ptr, 2, sdep, 0, 0);

      // task7
      ptr = __kmpc_omp_task_alloc(&loc, gtid, TIED, sizeof(task_t), 0, thunk_m);
      ptr->f_priv = t + 40; // init single first-private variable
      __kmpc_omp_task_with_deps(&loc, gtid, ptr, 2, sdep, 0, 0);
// compiler codegen end
      #pragma omp task depend(in: i3)
      { int th = omp_get_thread_num();
        printf("task 8_%d, th %d\n", t, th);
        #pragma omp atomic
          ++checker;
        if (checker != 1) { // last task should run exclusively
          err++;
          printf("Error1, checker %d != 1\n", checker); }
        mysleep(DELAY);
        if (checker != 1) { // last task should run exclusively
          err++;
          printf("Error1, checker %d != 1\n", checker); }
        #pragma omp atomic
          --checker;
      }
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
