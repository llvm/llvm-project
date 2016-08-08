// RUN: %libomp-compile -lpthread && %libomp-run
// The runtime currently does not get dependency information from GCC.
// UNSUPPORTED: gcc

#include <stdio.h>
#include <omp.h>
#include <pthread.h>
#include "omp_my_sleep.h"

/*
 With task dependencies one can generate proxy tasks from an explicit task
 being executed by a serial task team. The OpenMP runtime library didn't
 expect that and tries to free the explicit task that is the parent of the
 proxy task still working in background. It therefore has incomplete children
 which triggers a debugging assertion.
*/

// Compiler-generated code (emulation)
typedef long kmp_intptr_t;
typedef int kmp_int32;

typedef char bool;

typedef struct ident {
    kmp_int32 reserved_1;   /**<  might be used in Fortran; see above  */
    kmp_int32 flags;        /**<  also f.flags; KMP_IDENT_xxx flags; KMP_IDENT_KMPC identifies this union member  */
    kmp_int32 reserved_2;   /**<  not really used in Fortran any more; see above */
#if USE_ITT_BUILD
                            /*  but currently used for storing region-specific ITT */
                            /*  contextual information. */
#endif /* USE_ITT_BUILD */
    kmp_int32 reserved_3;   /**< source[4] in Fortran, do not use for C++  */
    char const *psource;    /**< String describing the source location.
                            The string is composed of semi-colon separated fields which describe the source file,
                            the function and a pair of line numbers that delimit the construct.
                             */
} ident_t;

typedef struct kmp_depend_info {
     kmp_intptr_t               base_addr;
     size_t                     len;
     struct {
         bool                   in:1;
         bool                   out:1;
     } flags;
} kmp_depend_info_t;

struct kmp_task;
typedef kmp_int32 (* kmp_routine_entry_t)( kmp_int32, struct kmp_task * );

typedef struct kmp_task {                   /* GEH: Shouldn't this be aligned somehow? */
    void *              shareds;            /**< pointer to block of pointers to shared vars   */
    kmp_routine_entry_t routine;            /**< pointer to routine to call for executing task */
    kmp_int32           part_id;            /**< part id for the task                          */
} kmp_task_t;

#ifdef __cplusplus
extern "C" {
#endif
kmp_int32  __kmpc_global_thread_num  ( ident_t * );
kmp_task_t*
__kmpc_omp_task_alloc( ident_t *loc_ref, kmp_int32 gtid, kmp_int32 flags,
                       size_t sizeof_kmp_task_t, size_t sizeof_shareds,
                       kmp_routine_entry_t task_entry );
void __kmpc_proxy_task_completed_ooo ( kmp_task_t *ptask );
kmp_int32 __kmpc_omp_task_with_deps ( ident_t *loc_ref, kmp_int32 gtid, kmp_task_t * new_task,
                                      kmp_int32 ndeps, kmp_depend_info_t *dep_list,
                                      kmp_int32 ndeps_noalias, kmp_depend_info_t *noalias_dep_list );
kmp_int32
__kmpc_omp_task( ident_t *loc_ref, kmp_int32 gtid, kmp_task_t * new_task );
#ifdef __cplusplus
}
#endif

void *target(void *task)
{
    my_sleep( 0.1 );
    __kmpc_proxy_task_completed_ooo((kmp_task_t*) task);
    return NULL;
}

pthread_t target_thread;

// User's code
int task_entry(kmp_int32 gtid, kmp_task_t *task)
{
    pthread_create(&target_thread, NULL, &target, task);
    return 0;
}

int main()
{
    int dep;

#pragma omp taskgroup
{
/*
 *  Corresponds to:
    #pragma omp target nowait depend(out: dep)
    {
        my_sleep( 0.1 );
    }
*/
    kmp_depend_info_t dep_info;
    dep_info.base_addr = (long) &dep;
    dep_info.len = sizeof(int);
    // out = inout per spec and runtime expects this
    dep_info.flags.in = 1;
    dep_info.flags.out = 1;

    kmp_int32 gtid = __kmpc_global_thread_num(NULL);
    kmp_task_t *proxy_task = __kmpc_omp_task_alloc(NULL,gtid,17,sizeof(kmp_task_t),0,&task_entry);
    __kmpc_omp_task_with_deps(NULL,gtid,proxy_task,1,&dep_info,0,NULL);

    #pragma omp task depend(in: dep)
    {
/*
 *      Corresponds to:
        #pragma omp target nowait depend(out: dep)
        {
            my_sleep( 0.1 );
        }
*/
        kmp_task_t *nested_proxy_task = __kmpc_omp_task_alloc(NULL,gtid,17,sizeof(kmp_task_t),0,&task_entry);
        __kmpc_omp_task(NULL,gtid,nested_proxy_task);
    }
}

    // only check that it didn't crash
    return 0;
}
