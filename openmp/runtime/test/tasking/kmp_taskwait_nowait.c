// RUN: %libomp-compile-and-run

// test checks IN dep kind in depend clause on taskwait nowait
// uses codegen emulation
// Note: no outlined task routine used
#include <stdio.h>
#include <omp.h>
// ---------------------------------------------------------------------------
// internal data to emulate compiler codegen
#define TIED 1
typedef struct DEP {
  size_t addr;
  size_t len;
  unsigned char flags;
} _dep;
typedef struct ID {
  int reserved_1;
  int flags;
  int reserved_2;
  int reserved_3;
  char *psource;
} _id;
typedef struct task {
  void** shareds;
  void* entry;
  int part_id;
  void* destr_thunk;
  int priority;
  long long device_id;
  int f_priv;
} task_t;
typedef int(*entry_t)(int, task_t*);

#ifdef __cplusplus
extern "C" {
#endif
extern int __kmpc_global_thread_num(_id*);
task_t *__kmpc_omp_task_alloc(_id *loc, int gtid, int flags,
                              size_t sz, size_t shar, entry_t rtn);
int __kmpc_omp_task_with_deps(_id *loc, int gtid, task_t *task, int ndeps,
                              _dep *dep_lst, int nd_noalias, _dep *noalias_l);
#ifdef __cplusplus
} // extern "C"
#endif

int main()
{
  int i1,i2,i3;
  omp_set_num_threads(2);
  printf("addresses: %p %p %p\n", &i1, &i2, &i3);
  #pragma omp parallel
  {
    int t = omp_get_thread_num();
    printf("thread %d enters parallel\n", t);
    #pragma omp single
    {
      #pragma omp task depend(in: i3)
      {
        int th = omp_get_thread_num();
        printf("task 0 created by th %d, executed by th %d\n", t, th);
      }
      #pragma omp task depend(in: i2)
      {
        int th = omp_get_thread_num();
        printf("task 1 created by th %d, executed by th %d\n", t, th);
      }
//      #pragma omp taskwait depend(in: i1, i2) nowait
      {
        _dep sdep[2];
        static _id loc = {0, 2, 0, 0, ";test.c;func;67;0;;"};
        int gtid = __kmpc_global_thread_num(&loc);
// instead of creating an empty task function we can now send NULL to runtime
        task_t *ptr = __kmpc_omp_task_alloc(&loc, gtid, TIED,
                                            sizeof(task_t), 0, NULL);
        sdep[0].addr = (size_t)&i2;
        sdep[0].flags = 1; // 1-in, 2-out, 3-inout, 4-mtx, 8-inoutset
        sdep[1].addr = (size_t)&i1;
        sdep[1].flags = 1; // in
        __kmpc_omp_task_with_deps(&loc, gtid, ptr, 2, sdep, 0, NULL);
      }
      printf("single done\n");
    }
  }
  printf("passed\n");
  return 0;
}
