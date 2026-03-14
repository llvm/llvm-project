// RUN: %libomp-compile-and-run
// RUN: %libomp-compile && env KMP_TASKLOOP_MIN_TASKS=1 %libomp-run

#include <stdio.h>
#include <omp.h>
#include "omp_my_sleep.h"

#define N 4
#define ST 3
#define UB 118
#define LB 0

// globals
int counter;
int task_count;

// Compiler-generated code (emulation)
typedef struct ident {
  void* dummy;
} ident_t;

typedef struct shar {
  int *pcounter;
  int *pj;
  int *ptask_count;
} *pshareds;

typedef struct task {
  pshareds shareds;
  int(* routine)(int,struct task*);
  int part_id;
  unsigned long long lb; // library always uses ULONG
  unsigned long long ub;
  int st;
  int last;
  int i;
  int j;
  int th;
} *ptask, kmp_task_t;

typedef int(* task_entry_t)( int, ptask );

void
__task_dup_entry(ptask task_dst, ptask task_src, int lastpriv)
{
// setup lastprivate flag
  task_dst->last = lastpriv;
// could be constructor calls here...
}

// OpenMP RTL interfaces
typedef unsigned long long kmp_uint64;
typedef long long kmp_int64;

#ifdef __cplusplus
extern "C" {
#endif
void
__kmpc_taskloop_5(ident_t *loc, int gtid, kmp_task_t *task, int if_val,
                  kmp_uint64 *lb, kmp_uint64 *ub, kmp_int64 st,
                  int nogroup, int sched, kmp_int64 grainsize, int modifier,
                  void *task_dup);
ptask
__kmpc_omp_task_alloc(ident_t *loc, int gtid, int flags,
                      size_t sizeof_kmp_task_t, size_t sizeof_shareds,
                      task_entry_t task_entry);
void __kmpc_atomic_fixed4_add(void *id_ref, int gtid, int * lhs, int rhs);
int  __kmpc_global_thread_num(void *id_ref);
#ifdef __cplusplus
}
#endif

// User's code
int task_entry(int gtid, ptask task)
{
  pshareds pshar = task->shareds;
  __kmpc_atomic_fixed4_add(NULL, gtid, pshar->ptask_count, 1);

  for (task->i = task->lb; task->i <= (int)task->ub; task->i += task->st) {
    task->th = omp_get_thread_num();
    __kmpc_atomic_fixed4_add(NULL,gtid,pshar->pcounter,1);
    task->j = task->i;
  }
  my_sleep( 0.1 ); // sleep 100 ms in order to allow other threads to steal tasks
  if (task->last) {
    *(pshar->pj) = task->j; // lastprivate
  }
  return 0;
}

void task_loop(int sched_type, int sched_val, int modifier)
{
  int i, j, gtid = __kmpc_global_thread_num(NULL);
  ptask task;
  pshareds psh;
  omp_set_dynamic(0);
  counter = 0;
  task_count = 0;
  #pragma omp parallel num_threads(N)
  {
    #pragma omp master
    {
      int gtid = __kmpc_global_thread_num(NULL);
      task = __kmpc_omp_task_alloc(NULL, gtid, 1, sizeof(struct task),
                                   sizeof(struct shar), &task_entry);
      psh = task->shareds;
      psh->pcounter = &counter;
      psh->ptask_count = &task_count;
      psh->pj = &j;
      task->lb = LB;
      task->ub = UB;
      task->st = ST;

      __kmpc_taskloop_5(
        NULL,             // location
        gtid,             // gtid
        task,             // task structure
        1,                // if clause value
        &task->lb,        // lower bound
        &task->ub,        // upper bound
        ST,               // loop increment
        0,                // 1 if nogroup specified
        sched_type,       // schedule type: 0-none, 1-grainsize, 2-num_tasks
        sched_val,        // schedule value (ignored for type 0)
        modifier,         // strict modifier
        (void*)&__task_dup_entry // tasks duplication routine
      );
    } // end master
  } // end parallel
// check results
  int tc;
  if (ST == 1) { // most common case
    tc = UB - LB + 1;
  } else if (ST < 0) {
    tc = (LB - UB) / (-ST) + 1;
  } else { // ST > 0
    tc = (UB - LB) / ST + 1;
  }
  int count;
  if (sched_type == 1) {
    count = (sched_val > tc) ? 1 : (tc + sched_val - 1) / sched_val;
  } else {
    count = (sched_val > tc) ? tc : sched_val;
  }
  if (j != LB + (tc - 1) * ST) {
    printf("Error in lastprivate, %d != %d\n", j, LB + (tc - 1) * ST);
    exit(1);
  }
  if (counter != tc) {
    printf("Error, counter %d != %d\n", counter, tc);
    exit(1);
  }
  if (task_count != count) {
    printf("Error, task count %d != %d\n", task_count, count);
    exit(1);
  }
}

int main(int argc, char *argv[]) {
  task_loop(1, 6, 1); // create 7 tasks
  task_loop(2, 6, 1); // create 6 tasks
  task_loop(1, 50, 1); // create 1 task
  task_loop(2, 50, 1); // create 40 tasks

  printf("Test passed\n");
  return 0;
}
