// RUN: %libomp-compile-and-run
// RUN: %libomp-compile && env KMP_TASKLOOP_MIN_TASKS=1 %libomp-run
// REQUIRES: openmp-4.5
#include <stdio.h>
#include <omp.h>
#include "omp_my_sleep.h"

#define N 4
#define GRAIN 10
#define STRIDE 3

// globals
int th_counter[N];
int counter;


// Compiler-generated code (emulation)
typedef struct ident {
    void* dummy;
} ident_t;

typedef struct shar {
    int(*pth_counter)[N];
    int *pcounter;
    int *pj;
} *pshareds;

typedef struct task {
    pshareds shareds;
    int(* routine)(int,struct task*);
    int part_id;
// privates:
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
__kmpc_taskloop(ident_t *loc, int gtid, kmp_task_t *task, int if_val,
                kmp_uint64 *lb, kmp_uint64 *ub, kmp_int64 st,
                int nogroup, int sched, kmp_int64 grainsize, void *task_dup );
ptask
__kmpc_omp_task_alloc( ident_t *loc, int gtid, int flags,
                  size_t sizeof_kmp_task_t, size_t sizeof_shareds,
                  task_entry_t task_entry );
void __kmpc_atomic_fixed4_add(void *id_ref, int gtid, int * lhs, int rhs);
int  __kmpc_global_thread_num(void *id_ref);
#ifdef __cplusplus
}
#endif


// User's code
int task_entry(int gtid, ptask task)
{
    pshareds pshar = task->shareds;
    for( task->i = task->lb; task->i <= (int)task->ub; task->i += task->st ) {
        task->th = omp_get_thread_num();
        __kmpc_atomic_fixed4_add(NULL,gtid,pshar->pcounter,1);
        __kmpc_atomic_fixed4_add(NULL,gtid,&((*pshar->pth_counter)[task->th]),1);
        task->j = task->i;
    }
    my_sleep( 0.1 ); // sleep 100 ms in order to allow other threads to steal tasks
    if( task->last ) {
        *(pshar->pj) = task->j; // lastprivate
    }
    return 0;
}

int main()
{
    int i, j, gtid = __kmpc_global_thread_num(NULL);
    ptask task;
    pshareds psh;
    omp_set_dynamic(0);
    counter = 0;
    for( i=0; i<N; ++i )
        th_counter[i] = 0;
    #pragma omp parallel num_threads(N)
    {
      #pragma omp master
      {
        int gtid = __kmpc_global_thread_num(NULL);
/*
 *  This is what the OpenMP runtime calls correspond to:
    #pragma omp taskloop num_tasks(N) lastprivate(j)
    for( i=0; i<N*GRAIN*STRIDE-1; i+=STRIDE )
    {
        int th = omp_get_thread_num();
        #pragma omp atomic
            counter++;
        #pragma omp atomic
            th_counter[th]++;
        j = i;
    }
*/
    task = __kmpc_omp_task_alloc(NULL,gtid,1,sizeof(struct task),sizeof(struct shar),&task_entry);
    psh = task->shareds;
    psh->pth_counter = &th_counter;
    psh->pcounter = &counter;
    psh->pj = &j;
    task->lb = 0;
    task->ub = N*GRAIN*STRIDE-2;
    task->st = STRIDE;

    __kmpc_taskloop(
        NULL,             // location
        gtid,             // gtid
        task,             // task structure
        1,                // if clause value
        &task->lb,        // lower bound
        &task->ub,        // upper bound
        STRIDE,           // loop increment
        0,                // 1 if nogroup specified
        2,                // schedule type: 0-none, 1-grainsize, 2-num_tasks
        N,                // schedule value (ignored for type 0)
        (void*)&__task_dup_entry // tasks duplication routine
        );
      } // end master
    } // end parallel
// check results
    if( j != N*GRAIN*STRIDE-STRIDE ) {
        printf("Error in lastprivate, %d != %d\n",j,N*GRAIN*STRIDE-STRIDE);
        return 1;
    }
    if( counter != N*GRAIN ) {
        printf("Error, counter %d != %d\n",counter,N*GRAIN);
        return 1;
    }
    for( i=0; i<N; ++i ) {
        if( th_counter[i] % GRAIN ) {
            printf("Error, th_counter[%d] = %d\n",i,th_counter[i]);
            return 1;
        }
    }
    printf("passed\n");
    return 0;
}
