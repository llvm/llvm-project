// RUN: %libomp-compile-and-run

// test checks IN dep kind in depend clause on taskwait construct
// uses codegen emulation
#include <stdio.h>
#include <omp.h>
// ---------------------------------------------------------------------------
// internal data to emulate compiler codegen
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

#ifdef __cplusplus
extern "C" {
#endif
extern int __kmpc_global_thread_num(_id*);
extern void __kmpc_omp_wait_deps(_id *, int, int, _dep *, int, _dep *);
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
//      #pragma omp taskwait depend(in: i1, i2)
      {
        _dep sdep[2];
        static _id loc = {0, 2, 0, 0, ";test9.c;func;60;0;;"};
        int gtid = __kmpc_global_thread_num(&loc);
        sdep[0].addr = (size_t)&i2;
        sdep[0].flags = 1; // 1-in, 2-out, 3-inout, 4-mtx, 8-inoutset
        sdep[1].addr = (size_t)&i1;
        sdep[1].flags = 1; // in
        __kmpc_omp_wait_deps(&loc, gtid, 2, sdep, 0, NULL);
      }
      printf("single done\n");
    }
  }
  printf("passed\n");
  return 0;
}
