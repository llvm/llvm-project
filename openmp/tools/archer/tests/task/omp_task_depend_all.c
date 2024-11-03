// RUN--: %libarcher-compile-and-run | FileCheck %s --check-prefix=NOENV
// RUN: %libarcher-compile && env ARCHER_OPTIONS="all_memory=1" \
// RUN:   %libarcher-run | FileCheck %s --check-prefix=ENV
// REQUIRES: tsan

// The runtime currently does not get dependency information from GCC.
// UNSUPPORTED: gcc

// Tests OMP 5.x task dependence "omp_all_memory",
// emulates compiler codegen versions for new dep kind
//
// Task tree created:
//      task0 - task1 (in: i1, i2)
//             \
//        task2 (inoutset: i2), (in: i1)
//             /
//        task3 (omp_all_memory) via flag=0x80
//             /
//      task4 - task5 (in: i1, i2)
//           /
//       task6 (omp_all_memory) via addr=-1
//           /
//       task7 (omp_all_memory) via flag=0x80
//           /
//       task8 (in: i3)
//

#include <omp.h>
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#define mysleep(n) Sleep(n)
#else
#include <unistd.h>
#define mysleep(n) usleep((n)*1000)
#endif

// to check the # of concurrent tasks (must be 1 for MTX, <3 for other kinds)
static int checker = 0;
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
#define DEP_ALL_MEM 0x80
typedef struct task {
  void **shareds;
  void *entry;
  int part_id;
  void *destr_thunk;
  int priority;
  long long device_id;
  int f_priv;
} task_t;
#define TIED 1
typedef int (*entry_t)(int, task_t *);
typedef struct ID {
  int reserved_1;
  int flags;
  int reserved_2;
  int reserved_3;
  char *psource;
} id;
// thunk routine for tasks with ALL dependency
int thunk_m(int gtid, task_t *ptask) {
  int lcheck, th;
#pragma omp atomic capture
  lcheck = ++checker;
  th = omp_get_thread_num();
  printf("task m_%d, th %d, checker %d\n", ptask->f_priv, th, lcheck);
  if (lcheck != 1) { // no more than 1 task at a time
    err++;
    printf("Error m1, checker %d != 1\n", lcheck);
  }
  mysleep(DELAY);
#pragma omp atomic read
  lcheck = checker; // must still be equal to 1
  if (lcheck != 1) {
    err++;
    printf("Error m2, checker %d != 1\n", lcheck);
  }
#pragma omp atomic
  --checker;
  return 0;
}
// thunk routine for tasks with inoutset dependency
int thunk_s(int gtid, task_t *ptask) {
  int lcheck, th;
#pragma omp atomic capture
  lcheck = ++checker; // 1
  th = omp_get_thread_num();
  printf("task 2_%d, th %d, checker %d\n", ptask->f_priv, th, lcheck);
  if (lcheck != 1) { // no more than 1 task at a time
    err++;
    printf("Error s1, checker %d != 1\n", lcheck);
  }
  mysleep(DELAY);
#pragma omp atomic read
  lcheck = checker; // must still be equal to 1
  if (lcheck != 1) {
    err++;
    printf("Error s2, checker %d != 1\n", lcheck);
  }
#pragma omp atomic
  --checker;
  return 0;
}

#ifdef __cplusplus
extern "C" {
#endif
int __kmpc_global_thread_num(id *);
task_t *__kmpc_omp_task_alloc(id *loc, int gtid, int flags, size_t sz,
                              size_t shar, entry_t rtn);
int __kmpc_omp_task_with_deps(id *loc, int gtid, task_t *task, int ndeps,
                              dep *dep_lst, int nd_noalias, dep *noalias_lst);
static id loc = {0, 2, 0, 0, ";file;func;0;0;;"};
#ifdef __cplusplus
} // extern "C"
#endif
// End of internal data
// ---------------------------------------------------------------------------

int main() {
  char *ompx_all_memory = (void *)0xffffffffffffffff;
  int i1, i2, i3;
  omp_set_num_threads(8);
  omp_set_dynamic(0);
#pragma omp parallel
  {
#pragma omp single nowait
    {
      dep sdep[2];
      task_t *ptr;
      int gtid = __kmpc_global_thread_num(&loc);
      int t = omp_get_thread_num();
#pragma omp task depend(in : i1, i2)
      { // task 0
        int lcheck, th;
#pragma omp atomic capture
        lcheck = ++checker; // 1 or 2
        th = omp_get_thread_num();
        printf("task 0_%d, th %d, checker %d\n", t, th, lcheck);
        if (lcheck > 2 || lcheck < 1) {
          err++; // no more than 2 tasks concurrently
          printf("Error1, checker %d, not 1 or 2\n", lcheck);
        }
        mysleep(DELAY);
#pragma omp atomic read
        lcheck = checker; // 1 or 2
        if (lcheck > 2 || lcheck < 1) {
#pragma omp atomic
          err++;
          printf("Error2, checker %d, not 1 or 2\n", lcheck);
        }
#pragma omp atomic
        --checker;
      }
#pragma omp task depend(in : i1, i2)
      { // task 1
        int lcheck, th;
#pragma omp atomic capture
        lcheck = ++checker; // 1 or 2
        th = omp_get_thread_num();
        printf("task 1_%d, th %d, checker %d\n", t, th, lcheck);
        if (lcheck > 2 || lcheck < 1) {
          err++; // no more than 2 tasks concurrently
          printf("Error3, checker %d, not 1 or 2\n", lcheck);
        }
        mysleep(DELAY);
#pragma omp atomic read
        lcheck = checker; // 1 or 2
        if (lcheck > 2 || lcheck < 1) {
          err++;
          printf("Error4, checker %d, not 1 or 2\n", lcheck);
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
      sdep[1].len = 0;      // not used
      sdep[1].flags = 8;    // INOUTSET
      ptr->f_priv = t + 10; // init single first-private variable
      __kmpc_omp_task_with_deps(&loc, gtid, ptr, 2, sdep, 0, 0);

// task3
#pragma omp task depend(in : i1) depend(inout : ompx_all_memory[0])
      {
        int lcheck, th;
#pragma omp atomic capture
        lcheck = ++checker;
        th = omp_get_thread_num();
        printf("task 3_%d, th %d, checker %d\n", t, th, lcheck);
        if (lcheck != 1) { // no more than 1 task at a time
          err++;
          printf("Error m1, checker %d != 1\n", lcheck);
        }
        mysleep(DELAY);
#pragma omp atomic read
        lcheck = checker; // must still be equal to 1
        if (lcheck != 1) {
          err++;
          printf("Error m2, checker %d != 1\n", lcheck);
        }
#pragma omp atomic
        --checker;
      }
      // compiler codegen end
#pragma omp task depend(in : i1, i2)
      { // task 4
        int lcheck, th;
#pragma omp atomic capture
        lcheck = ++checker; // 1 or 2
        th = omp_get_thread_num();
        printf("task 4_%d, th %d, checker %d\n", t, th, lcheck);
        if (lcheck > 2 || lcheck < 1) {
          err++; // no more than 2 tasks concurrently
          printf("Error5, checker %d, not 1 or 2\n", lcheck);
        }
        mysleep(DELAY);
#pragma omp atomic read
        lcheck = checker; // 1 or 2
        if (lcheck > 2 || lcheck < 1) {
          err++;
          printf("Error6, checker %d, not 1 or 2\n", lcheck);
        }
#pragma omp atomic
        --checker;
      }
#pragma omp task depend(in : i1, i2)
      { // task 5
        int lcheck, th;
#pragma omp atomic capture
        lcheck = ++checker; // 1 or 2
        th = omp_get_thread_num();
        printf("task 5_%d, th %d, checker %d\n", t, th, lcheck);
        if (lcheck > 2 || lcheck < 1) {
          err++; // no more than 2 tasks concurrently
          printf("Error7, checker %d, not 1 or 2\n", lcheck);
        }
        mysleep(DELAY);
#pragma omp atomic read
        lcheck = checker; // 1 or 2
        if (lcheck > 2 || lcheck < 1) {
          err++;
          printf("Error8, checker %d, not 1 or 2\n", lcheck);
        }
#pragma omp atomic
        --checker;
      }
// task6
#pragma omp task depend(inout : ompx_all_memory[0])
      {
        int lcheck, th;
#pragma omp atomic capture
        lcheck = ++checker;
        th = omp_get_thread_num();
        printf("task 6_%d, th %d, checker %d\n", t, th, lcheck);
        if (lcheck != 1) { // no more than 1 task at a time
          err++;
          printf("Error m1, checker %d != 1\n", lcheck);
        }
        mysleep(DELAY);
#pragma omp atomic read
        lcheck = checker; // must still be equal to 1
        if (lcheck != 1) {
          err++;
          printf("Error m2, checker %d != 1\n", lcheck);
        }
#pragma omp atomic
        --checker;
      }
// task7
#pragma omp task depend(inout : ompx_all_memory[0]) depend(mutexinoutset : i3)
      {
        int lcheck, th;
#pragma omp atomic capture
        lcheck = ++checker;
        th = omp_get_thread_num();
        printf("task 7_%d, th %d, checker %d\n", t, th, lcheck);
        if (lcheck != 1) { // no more than 1 task at a time
          err++;
          printf("Error m1, checker %d != 1\n", lcheck);
        }
        mysleep(DELAY);
#pragma omp atomic read
        lcheck = checker; // must still be equal to 1
        if (lcheck != 1) {
          err++;
          printf("Error m2, checker %d != 1\n", lcheck);
        }
#pragma omp atomic
        --checker;
      }
#pragma omp task depend(in : i3)
      { // task 8
        int lcheck, th;
#pragma omp atomic capture
        lcheck = ++checker; // 1
        th = omp_get_thread_num();
        printf("task 8_%d, th %d, checker %d\n", t, th, lcheck);
        if (lcheck != 1) {
          err++;
          printf("Error9, checker %d, != 1\n", lcheck);
        }
        mysleep(DELAY);
#pragma omp atomic read
        lcheck = checker;
        if (lcheck != 1) {
          err++;
          printf("Error10, checker %d, != 1\n", lcheck);
        }
#pragma omp atomic
        --checker;
      }
    } // single
  }   // parallel
  if (err == 0 && checker == 0) {
    printf("passed\n");
    return 0;
  } else {
    printf("failed, err = %d, checker = %d\n", err, checker);
    return 1;
  }
}

// NOENV-NOT: ThreadSanitizer: data race
// NOENV-NOT: ThreadSanitizer: reported
// NOENV: omp_all_memory
// NOENV-NOT: ThreadSanitizer: data race
// NOENV-NOT: ThreadSanitizer: reported
// NOENV: passed

// ENV-NOT: ThreadSanitizer: data race
// ENV-NOT: ThreadSanitizer: reported
// ENV: passed
