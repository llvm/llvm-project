// RUN: %libomp-compile && env OMP_NUM_THREADS='3' %libomp-run
// RUN: %libomp-compile && env OMP_NUM_THREADS='1' %libomp-run

#include <stdio.h>
#include <omp.h>
#include "omp_my_sleep.h"

// detached tied
#define PTASK_FLAG_DETACHABLE 0x41

// OpenMP RTL interfaces
typedef unsigned long long kmp_uint64;
typedef long long kmp_int64;

typedef struct ID {
  int reserved_1;
  int flags;
  int reserved_2;
  int reserved_3;
  char *psource;
} id;

// Compiler-generated code (emulation)
typedef struct ident {
  void* dummy; // not used in the library
} ident_t;

typedef enum kmp_event_type_t {
  KMP_EVENT_UNINITIALIZED = 0,
  KMP_EVENT_ALLOW_COMPLETION = 1
} kmp_event_type_t;

typedef struct {
  kmp_event_type_t type;
  union {
    void *task;
  } ed;
} kmp_event_t;

typedef struct shar { // shareds used in the task
} *pshareds;

typedef struct task {
  pshareds shareds;
  int(*routine)(int,struct task*);
  int part_id;
// void *destructor_thunk; // optional, needs flag setting if provided
// int priority; // optional, needs flag setting if provided
// ------------------------------
// privates used in the task:
  omp_event_handle_t evt;
} *ptask, kmp_task_t;

typedef int(* task_entry_t)( int, ptask );

#ifdef __cplusplus
extern "C" {
#endif
extern int  __kmpc_global_thread_num(void *id_ref);
extern int** __kmpc_omp_task_alloc(id *loc, int gtid, int flags,
                                   size_t sz, size_t shar, task_entry_t rtn);
extern int __kmpc_omp_task(id *loc, int gtid, kmp_task_t *task);
extern omp_event_handle_t __kmpc_task_allow_completion_event(
                              ident_t *loc_ref, int gtid, kmp_task_t *task);
#ifdef __cplusplus
}
#endif

int volatile checker;

// User's code, outlined into task entry
int task_entry(int gtid, ptask task) {
  my_sleep(2.0);
  checker = 1;
  return 0;
}

int main() {
  int i, j, gtid = __kmpc_global_thread_num(NULL);
  int nt = omp_get_max_threads();
  ptask task;
  pshareds psh;
  checker = 0;
  omp_set_dynamic(0);
  #pragma omp parallel //num_threads(N)
  {
    #pragma omp master
    {
      int gtid = __kmpc_global_thread_num(NULL);
      omp_event_handle_t evt;
/*
      #pragma omp task detach(evt)
      {}
*/
      task = (ptask)__kmpc_omp_task_alloc(NULL,gtid,PTASK_FLAG_DETACHABLE,
                        sizeof(struct task),sizeof(struct shar),&task_entry);
      psh = task->shareds;
      evt = (omp_event_handle_t)__kmpc_task_allow_completion_event(NULL,gtid,task);
      task->evt = evt;
      __kmpc_omp_task(NULL, gtid, task);
      omp_fulfill_event(evt);
      #pragma omp taskwait
      ;
//      printf("after tw %d\n", omp_get_thread_num());
    } // end master
  } // end parallel

  // check results
  if (checker == 1) {
    printf("passed\n");
    return 0;
  } else {
    printf("failed\n");
    return 1;
  }
}
