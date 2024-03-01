// RUN: %libomp-compile-and-run
// The runtime currently does not get dependency information from GCC.
// UNSUPPORTED: gcc

// Very flaky on openmp-clang-x86_64-linux-debian.
// https://bugs.llvm.org/show_bug.cgi?id=45397
// UNSUPPORTED: linux

#include <stdint.h>
#include <stdio.h>
#include <omp.h>
#include "omp_testsuite.h"
#include "omp_my_sleep.h"

/*
 An explicit task can have a dependency on a target task. If it is not
 directly satisfied, the runtime should not wait but resume execution.
*/

// Compiler-generated code (emulation)
typedef intptr_t kmp_intptr_t;
typedef int32_t kmp_int32;
typedef uint8_t kmp_uint8;

typedef char bool;

// These structs need to match the implementation within libomp, in kmp.h.

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
     union {
        kmp_uint8 flag; // flag as an unsigned char
        struct { // flag as a set of 8 bits
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
          unsigned all : 1;
          unsigned unused : 3;
          unsigned set : 1;
          unsigned mtx : 1;
          unsigned out : 1;
          unsigned in : 1;
#else
          unsigned in : 1;
          unsigned out : 1;
          unsigned mtx : 1;
          unsigned set : 1;
          unsigned unused : 3;
          unsigned all : 1;
#endif
        } flags;
    };
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

/*
 *  Corresponds to:
    #pragma omp target nowait depend(out: dep)
    {
        my_sleep( 0.1 );
    }
*/
    kmp_depend_info_t dep_info = { 0 };
    dep_info.base_addr = (kmp_intptr_t) &dep;
    dep_info.len = sizeof(int);
    // out = inout per spec and runtime expects this
    dep_info.flags.in = 1;
    dep_info.flags.out = 1;

    kmp_int32 gtid = __kmpc_global_thread_num(NULL);
    kmp_task_t *proxy_task = __kmpc_omp_task_alloc(NULL,gtid,17,sizeof(kmp_task_t),0,&task_entry);
    __kmpc_omp_task_with_deps(NULL,gtid,proxy_task,1,&dep_info,0,NULL);

    int first_task_finished = 0;
    #pragma omp task shared(first_task_finished) depend(inout: dep)
    {
        first_task_finished = 1;
    }

    int second_task_finished = 0;
    #pragma omp task shared(second_task_finished) depend(in: dep)
    {
        second_task_finished = 1;
    }

    // check that execution has been resumed and the runtime has not waited
    // for the dependencies to be satisfied.
    int error = (first_task_finished == 1);
    error += (second_task_finished == 1);

    #pragma omp taskwait

    // by now all tasks should have finished
    error += (first_task_finished != 1);
    error += (second_task_finished != 1);

    return error;
}
