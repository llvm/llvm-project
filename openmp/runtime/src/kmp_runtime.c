/*
 * kmp_runtime.c -- KPTS runtime support library
 * $Revision: 42839 $
 * $Date: 2013-11-24 13:01:00 -0600 (Sun, 24 Nov 2013) $
 */


//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


#include "kmp.h"
#include "kmp_atomic.h"
#include "kmp_wrapper_getpid.h"
#include "kmp_environment.h"
#include "kmp_itt.h"
#include "kmp_str.h"
#include "kmp_settings.h"
#include "kmp_i18n.h"
#include "kmp_io.h"
#include "kmp_error.h"

/* these are temporary issues to be dealt with */
#define KMP_USE_PRCTL 0
#define KMP_USE_POOLED_ALLOC 0

#if KMP_MIC
#include <immintrin.h>
#define USE_NGO_STORES 1
#endif // KMP_MIC

#if KMP_MIC && USE_NGO_STORES
#define load_icvs(src)         __m512d Vt_icvs = _mm512_load_pd((void *)(src))
#define store_icvs(dst, src)   _mm512_storenrngo_pd((void *)(dst), Vt_icvs)
#define sync_icvs()            __asm__ volatile ("lock; addl $0,0(%%rsp)" ::: "memory")
#else
#define load_icvs(src)         ((void)0)
#define store_icvs(dst, src)   copy_icvs((dst), (src))
#define sync_icvs()            ((void)0)
#endif /* KMP_MIC && USE_NGO_STORES */

#if KMP_OS_WINDOWS
#include <process.h>
#endif


#if defined(KMP_GOMP_COMPAT)
char const __kmp_version_alt_comp[] = KMP_VERSION_PREFIX "alternative compiler support: yes";
#endif /* defined(KMP_GOMP_COMPAT) */

char const __kmp_version_omp_api[] = KMP_VERSION_PREFIX "API version: "
#if OMP_40_ENABLED
    "4.0 (201307)";
#elif OMP_30_ENABLED
    "3.1 (201107)";
#else
    "2.5 (200505)";
#endif

#ifdef KMP_DEBUG

char const __kmp_version_lock[] = KMP_VERSION_PREFIX "lock type: run time selectable";

char const __kmp_version_perf_v19[] = KMP_VERSION_PREFIX "perf v19: "
#if KMP_PERF_V19 == KMP_ON
    "on";
#elif KMP_PERF_V19 == KMP_OFF
    "off";
#else
    #error "Must specify KMP_PERF_V19 option"
#endif

char const __kmp_version_perf_v106[] = KMP_VERSION_PREFIX "perf v106: "
#if KMP_PERF_V106 == KMP_ON
    "on";
#elif KMP_PERF_V106 == KMP_OFF
    "off";
#else
    #error "Must specify KMP_PERF_V106 option"
#endif

#endif /* KMP_DEBUG */


#define KMP_MIN( x, y ) ( (x) < (y) ? (x) : (y) )

/* ------------------------------------------------------------------------ */
/* ------------------------------------------------------------------------ */

kmp_info_t __kmp_monitor;

/* ------------------------------------------------------------------------ */
/* ------------------------------------------------------------------------ */

/* Forward declarations */

void __kmp_cleanup( void );

static void __kmp_initialize_info( kmp_info_t *, kmp_team_t *, int tid, int gtid );
static void __kmp_initialize_team(
    kmp_team_t * team,
    int          new_nproc,
    #if OMP_30_ENABLED
        kmp_internal_control_t * new_icvs,
        ident_t *                loc
    #else
        int new_set_nproc, int new_set_dynamic, int new_set_nested,
        int new_set_blocktime, int new_bt_intervals, int new_bt_set
    #endif // OMP_30_ENABLED
);
static void __kmp_partition_places( kmp_team_t *team );
static void __kmp_do_serial_initialize( void );


#ifdef USE_LOAD_BALANCE
static int __kmp_load_balance_nproc( kmp_root_t * root, int set_nproc );
#endif

static int __kmp_expand_threads(int nWish, int nNeed);
static int __kmp_unregister_root_other_thread( int gtid );
static void __kmp_unregister_library( void ); // called by __kmp_internal_end()
static void __kmp_reap_thread( kmp_info_t * thread, int is_root );
static kmp_info_t *__kmp_thread_pool_insert_pt = NULL;

/* ------------------------------------------------------------------------ */
/* ------------------------------------------------------------------------ */

/* Calculate the identifier of the current thread */
/* fast (and somewhat portable) way to get unique */
/* identifier of executing thread.                */
/* returns KMP_GTID_DNE if we haven't been assigned a gtid   */

int
__kmp_get_global_thread_id( )
{
    int i;
    kmp_info_t   **other_threads;
    size_t         stack_data;
    char          *stack_addr;
    size_t         stack_size;
    char          *stack_base;

    KA_TRACE( 1000, ( "*** __kmp_get_global_thread_id: entering, nproc=%d  all_nproc=%d\n",
                      __kmp_nth, __kmp_all_nth ));

    /* JPH - to handle the case where __kmpc_end(0) is called immediately prior to a
             parallel region, made it return KMP_GTID_DNE to force serial_initialize by
             caller.  Had to handle KMP_GTID_DNE at all call-sites, or else guarantee
             __kmp_init_gtid for this to work.  */

    if ( !TCR_4(__kmp_init_gtid) ) return KMP_GTID_DNE;

#ifdef KMP_TDATA_GTID
    if ( TCR_4(__kmp_gtid_mode) >= 3) {
        KA_TRACE( 1000, ( "*** __kmp_get_global_thread_id: using TDATA\n" ));
        return __kmp_gtid;
    }
#endif
    if ( TCR_4(__kmp_gtid_mode) >= 2) {
        KA_TRACE( 1000, ( "*** __kmp_get_global_thread_id: using keyed TLS\n" ));
        return __kmp_gtid_get_specific();
    }
    KA_TRACE( 1000, ( "*** __kmp_get_global_thread_id: using internal alg.\n" ));

    stack_addr    = (char*) & stack_data;
    other_threads = __kmp_threads;

    /*
        ATT: The code below is a source of potential bugs due to unsynchronized access to
        __kmp_threads array. For example:
            1. Current thread loads other_threads[i] to thr and checks it, it is non-NULL.
            2. Current thread is suspended by OS.
            3. Another thread unregisters and finishes (debug versions of free() may fill memory
               with something like 0xEF).
            4. Current thread is resumed.
            5. Current thread reads junk from *thr.
        TODO: Fix it.
        --ln
    */

    for( i = 0 ; i < __kmp_threads_capacity ; i++ ) {

        kmp_info_t *thr = (kmp_info_t *)TCR_SYNC_PTR(other_threads[i]);
        if( !thr ) continue;

        stack_size =  (size_t)TCR_PTR(thr -> th.th_info.ds.ds_stacksize);
        stack_base =  (char *)TCR_PTR(thr -> th.th_info.ds.ds_stackbase);

        /* stack grows down -- search through all of the active threads */

        if( stack_addr <= stack_base ) {
            size_t stack_diff = stack_base - stack_addr;

            if( stack_diff <= stack_size ) {
                /* The only way we can be closer than the allocated */
                /* stack size is if we are running on this thread. */
                KMP_DEBUG_ASSERT( __kmp_gtid_get_specific() == i );
                return i;
            }
        }
    }

    /* get specific to try and determine our gtid */
    KA_TRACE( 1000, ( "*** __kmp_get_global_thread_id: internal alg. failed to find "
                      "thread, using TLS\n" ));
    i = __kmp_gtid_get_specific();

    /*fprintf( stderr, "=== %d\n", i );  */ /* GROO */

    /* if we havn't been assigned a gtid, then return code */
    if( i<0 ) return i;

    /* dynamically updated stack window for uber threads to avoid get_specific call */
    if( ! TCR_4(other_threads[i]->th.th_info.ds.ds_stackgrow) ) {
        KMP_FATAL( StackOverflow, i );
    }

    stack_base = (char *) other_threads[i] -> th.th_info.ds.ds_stackbase;
    if( stack_addr > stack_base ) {
        TCW_PTR(other_threads[i]->th.th_info.ds.ds_stackbase, stack_addr);
        TCW_PTR(other_threads[i]->th.th_info.ds.ds_stacksize,
          other_threads[i]->th.th_info.ds.ds_stacksize + stack_addr - stack_base);
    } else {
        TCW_PTR(other_threads[i]->th.th_info.ds.ds_stacksize, stack_base - stack_addr);
    }

    /* Reprint stack bounds for ubermaster since they have been refined */
    if ( __kmp_storage_map ) {
        char *stack_end = (char *) other_threads[i] -> th.th_info.ds.ds_stackbase;
        char *stack_beg = stack_end - other_threads[i] -> th.th_info.ds.ds_stacksize;
        __kmp_print_storage_map_gtid( i, stack_beg, stack_end,
                                      other_threads[i] -> th.th_info.ds.ds_stacksize,
                                      "th_%d stack (refinement)", i );
    }
    return i;
}

int
__kmp_get_global_thread_id_reg( )
{
    int gtid;

    if ( !__kmp_init_serial ) {
        gtid = KMP_GTID_DNE;
    } else
#ifdef KMP_TDATA_GTID
    if ( TCR_4(__kmp_gtid_mode) >= 3 ) {
        KA_TRACE( 1000, ( "*** __kmp_get_global_thread_id_reg: using TDATA\n" ));
        gtid = __kmp_gtid;
    } else
#endif
    if ( TCR_4(__kmp_gtid_mode) >= 2 ) {
        KA_TRACE( 1000, ( "*** __kmp_get_global_thread_id_reg: using keyed TLS\n" ));
        gtid = __kmp_gtid_get_specific();
    } else {
        KA_TRACE( 1000, ( "*** __kmp_get_global_thread_id_reg: using internal alg.\n" ));
        gtid = __kmp_get_global_thread_id();
    }

    /* we must be a new uber master sibling thread */
    if( gtid == KMP_GTID_DNE ) {
        KA_TRACE( 10, ( "__kmp_get_global_thread_id_reg: Encountered new root thread. "
                        "Registering a new gtid.\n" ));
        __kmp_acquire_bootstrap_lock( &__kmp_initz_lock );
        if( !__kmp_init_serial ) {
            __kmp_do_serial_initialize();
            gtid = __kmp_gtid_get_specific();
        } else {
            gtid = __kmp_register_root(FALSE);
        }
        __kmp_release_bootstrap_lock( &__kmp_initz_lock );
        /*__kmp_printf( "+++ %d\n", gtid ); */ /* GROO */
    }

    KMP_DEBUG_ASSERT( gtid >=0 );

    return gtid;
}

/* caller must hold forkjoin_lock */
void
__kmp_check_stack_overlap( kmp_info_t *th )
{
    int f;
    char *stack_beg = NULL;
    char *stack_end = NULL;
    int gtid;

    KA_TRACE(10,("__kmp_check_stack_overlap: called\n"));
    if ( __kmp_storage_map ) {
        stack_end = (char *) th -> th.th_info.ds.ds_stackbase;
        stack_beg = stack_end - th -> th.th_info.ds.ds_stacksize;

        gtid = __kmp_gtid_from_thread( th );

        if (gtid == KMP_GTID_MONITOR) {
            __kmp_print_storage_map_gtid( gtid, stack_beg, stack_end, th->th.th_info.ds.ds_stacksize,
                                     "th_%s stack (%s)", "mon",
                                     ( th->th.th_info.ds.ds_stackgrow ) ? "initial" : "actual" );
        } else {
            __kmp_print_storage_map_gtid( gtid, stack_beg, stack_end, th->th.th_info.ds.ds_stacksize,
                                     "th_%d stack (%s)", gtid,
                                     ( th->th.th_info.ds.ds_stackgrow ) ? "initial" : "actual" );
        }
    }

    /* No point in checking ubermaster threads since they use refinement and cannot overlap */
    if ( __kmp_env_checks == TRUE && !KMP_UBER_GTID(gtid = __kmp_gtid_from_thread( th )))
    {
        KA_TRACE(10,("__kmp_check_stack_overlap: performing extensive checking\n"));
        if ( stack_beg == NULL ) {
            stack_end = (char *) th -> th.th_info.ds.ds_stackbase;
            stack_beg = stack_end - th -> th.th_info.ds.ds_stacksize;
        }

        for( f=0 ; f < __kmp_threads_capacity ; f++ ) {
            kmp_info_t *f_th = (kmp_info_t *)TCR_SYNC_PTR(__kmp_threads[f]);

            if( f_th && f_th != th ) {
                char *other_stack_end = (char *)TCR_PTR(f_th->th.th_info.ds.ds_stackbase);
                char *other_stack_beg = other_stack_end -
                                        (size_t)TCR_PTR(f_th->th.th_info.ds.ds_stacksize);
                if((stack_beg > other_stack_beg && stack_beg < other_stack_end) ||
                   (stack_end > other_stack_beg && stack_end < other_stack_end)) {

                    /* Print the other stack values before the abort */
                    if ( __kmp_storage_map )
                        __kmp_print_storage_map_gtid( -1, other_stack_beg, other_stack_end,
                            (size_t)TCR_PTR(f_th->th.th_info.ds.ds_stacksize),
                            "th_%d stack (overlapped)",
                                                 __kmp_gtid_from_thread( f_th ) );

                    __kmp_msg( kmp_ms_fatal, KMP_MSG( StackOverlap ), KMP_HNT( ChangeStackLimit ), __kmp_msg_null );
                }
            }
        }
    }
    KA_TRACE(10,("__kmp_check_stack_overlap: returning\n"));
}


/* ------------------------------------------------------------------------ */

#ifndef KMP_DEBUG
# define __kmp_static_delay( arg )     /* nothing to do */
#else

static void
__kmp_static_delay( int arg )
{
/* Work around weird code-gen bug that causes assert to trip */
# if KMP_ARCH_X86_64 && KMP_OS_LINUX
    KMP_ASSERT( arg != 0 );
# else
    KMP_ASSERT( arg >= 0 );
# endif
}
#endif /* KMP_DEBUG */

static void
__kmp_static_yield( int arg )
{
    __kmp_yield( arg );
}

/*
 * Spin wait loop that first does pause, then yield, then sleep.
 * Wait until spinner is equal to checker to exit.
 *
 * A thread that calls __kmp_wait_sleep must make certain that another thread
 * calls __kmp_release to wake it back up up to prevent deadlocks!
 */

void
__kmp_wait_sleep( kmp_info_t *this_thr,
                  volatile kmp_uint *spinner,
                  kmp_uint checker,
                  int final_spin
                  USE_ITT_BUILD_ARG (void * itt_sync_obj)
)
{
    /* note: we may not belong to a team at this point */
    register volatile kmp_uint    *spin      = spinner;
    register          kmp_uint     check     = checker;
    register          kmp_uint32   spins;
    register          kmp_uint32   hibernate;
                      int          th_gtid, th_tid;
#if OMP_30_ENABLED
                      int          flag = FALSE;
#endif /* OMP_30_ENABLED */

    KMP_FSYNC_SPIN_INIT( spin, NULL );
    if( TCR_4(*spin) == check ) {
        KMP_FSYNC_SPIN_ACQUIRED( spin );
        return;
    }

    th_gtid = this_thr->th.th_info.ds.ds_gtid;

    KA_TRACE( 20, ("__kmp_wait_sleep: T#%d waiting for spin(%p) == %d\n",
                  th_gtid,
                  spin, check ) );

    /* setup for waiting */
    KMP_INIT_YIELD( spins );

    if ( __kmp_dflt_blocktime != KMP_MAX_BLOCKTIME ) {
        //
        // The worker threads cannot rely on the team struct existing at this
        // point.  Use the bt values cached in the thread struct instead.
        //
 #ifdef KMP_ADJUST_BLOCKTIME
        if ( __kmp_zero_bt && ! this_thr->th.th_team_bt_set ) {
            /* force immediate suspend if not set by user and more threads than available procs */
            hibernate = 0;
        } else {
            hibernate = this_thr->th.th_team_bt_intervals;
        }
 #else
        hibernate = this_thr->th.th_team_bt_intervals;
 #endif /* KMP_ADJUST_BLOCKTIME */

        //
        // If the blocktime is nonzero, we want to make sure that we spin
        // wait for the entirety of the specified #intervals, plus up to
        // one interval more.  This increment make certain that this thread
        // doesn't go to sleep too soon.
        //
        if ( hibernate != 0 ) {
            hibernate++;
        }

        //
        // Add in the current time value.
        //
        hibernate += TCR_4( __kmp_global.g.g_time.dt.t_value );

        KF_TRACE( 20, ("__kmp_wait_sleep: T#%d now=%d, hibernate=%d, intervals=%d\n",
                       th_gtid, __kmp_global.g.g_time.dt.t_value, hibernate,
                       hibernate - __kmp_global.g.g_time.dt.t_value ));
    }

    KMP_MB();

    /* main wait spin loop */
    while( TCR_4(*spin) != check ) {
        int in_pool;

#if OMP_30_ENABLED
        //
        // If the task team is NULL, it means one of things:
        //   1) A newly-created thread is first being released by
        //      __kmp_fork_barrier(), and its task team has not been set up
        //      yet.
        //   2) All tasks have been executed to completion, this thread has
        //      decremented the task team's ref ct and possibly deallocated
        //      it, and should no longer reference it.
        //   3) Tasking is off for this region.  This could be because we
        //      are in a serialized region (perhaps the outer one), or else
        //      tasking was manually disabled (KMP_TASKING=0).
        //
        kmp_task_team_t * task_team = NULL;
        if ( __kmp_tasking_mode != tskm_immediate_exec ) {
            task_team = this_thr->th.th_task_team;
            if ( task_team != NULL ) {
                if ( ! TCR_SYNC_4( task_team->tt.tt_active ) ) {
                    KMP_DEBUG_ASSERT( ! KMP_MASTER_TID( this_thr->th.th_info.ds.ds_tid ) );
                    __kmp_unref_task_team( task_team, this_thr );
                } else if ( KMP_TASKING_ENABLED( task_team, this_thr->th.th_task_state ) ) {
                    __kmp_execute_tasks( this_thr, th_gtid, spin, check, final_spin, &flag
                                         USE_ITT_BUILD_ARG( itt_sync_obj ), 0);
                }
            }; // if
        }; // if
#endif /* OMP_30_ENABLED */

        KMP_FSYNC_SPIN_PREPARE( spin );
        if( TCR_4(__kmp_global.g.g_done) ) {
            if( __kmp_global.g.g_abort )
                __kmp_abort_thread( );
            break;
        }

        __kmp_static_delay( 1 );

        /* if we are oversubscribed,
           or have waited a bit (and KMP_LIBRARY=throughput), then yield */
        KMP_YIELD( TCR_4(__kmp_nth) > __kmp_avail_proc );
        // TODO: Should it be number of cores instead of thread contexts? Like:
        // KMP_YIELD( TCR_4(__kmp_nth) > __kmp_ncores );
        // Need performance improvement data to make the change...
        KMP_YIELD_SPIN( spins );

        //
        // Check if this thread was transferred from a team
        // to the thread pool (or vice-versa) while spinning.
        //
        in_pool = !!TCR_4(this_thr->th.th_in_pool);
        if ( in_pool != !!this_thr->th.th_active_in_pool ) {
            if ( in_pool ) {
                //
                // recently transferred from team to pool
                //
                KMP_TEST_THEN_INC32(
                                    (kmp_int32 *) &__kmp_thread_pool_active_nth );
                this_thr->th.th_active_in_pool = TRUE;

                //
                // Here, we cannot assert that
                //
                // KMP_DEBUG_ASSERT( TCR_4(__kmp_thread_pool_active_nth)
                //  <= __kmp_thread_pool_nth );
                //
                // __kmp_thread_pool_nth is inc/dec'd by the master thread
                // while the fork/join lock is held, whereas
                // __kmp_thread_pool_active_nth is inc/dec'd asynchronously
                // by the workers.  The two can get out of sync for brief
                // periods of time.
                //
            }
            else {
                //
                // recently transferred from pool to team
                //
                KMP_TEST_THEN_DEC32(
                                    (kmp_int32 *) &__kmp_thread_pool_active_nth );
                KMP_DEBUG_ASSERT( TCR_4(__kmp_thread_pool_active_nth) >= 0 );
                this_thr->th.th_active_in_pool = FALSE;
            }
        }

#if OMP_30_ENABLED
        // Don't suspend if there is a likelihood of new tasks being spawned.
        if ( ( task_team != NULL ) && TCR_4(task_team->tt.tt_found_tasks) ) {
            continue;
        }
#endif /* OMP_30_ENABLED */

        /* Don't suspend if KMP_BLOCKTIME is set to "infinite" */
        if ( __kmp_dflt_blocktime == KMP_MAX_BLOCKTIME ) {
            continue;
        }

        /* if we have waited a bit more, fall asleep */
        if ( TCR_4( __kmp_global.g.g_time.dt.t_value ) < hibernate ) {
            continue;
        }

        KF_TRACE( 50, ("__kmp_wait_sleep: T#%d suspend time reached\n", th_gtid ) );

        __kmp_suspend( th_gtid, spin, check );

        if( TCR_4( __kmp_global.g.g_done ) && __kmp_global.g.g_abort ) {
            __kmp_abort_thread( );
        }

        /* TODO */
        /* if thread is done with work and timesout, disband/free */
    }

    KMP_FSYNC_SPIN_ACQUIRED( spin );
}


/*
 * Release the thread specified by target_thr from waiting by setting the location
 * specified by spin and resume the thread if indicated by the sleep parameter.
 *
 * A thread that calls __kmp_wait_sleep must call this function to wake up the
 * potentially sleeping thread and prevent deadlocks!
 */

void
__kmp_release( kmp_info_t *target_thr, volatile kmp_uint *spin,
               enum kmp_mem_fence_type fetchadd_fence )
{
    kmp_uint old_spin;
    #ifdef KMP_DEBUG
        int target_gtid = target_thr->th.th_info.ds.ds_gtid;
        int gtid = TCR_4(__kmp_init_gtid) ? __kmp_get_gtid() : -1;
    #endif

    KF_TRACE( 20, ( "__kmp_release: T#%d releasing T#%d spin(%p) fence_type(%d)\n",
                    gtid, target_gtid, spin, fetchadd_fence ));

    KMP_DEBUG_ASSERT( spin );

    KMP_DEBUG_ASSERT( fetchadd_fence == kmp_acquire_fence ||
                      fetchadd_fence == kmp_release_fence );

    KMP_FSYNC_RELEASING( spin );

    old_spin = ( fetchadd_fence == kmp_acquire_fence )
                 ? KMP_TEST_THEN_ADD4_ACQ32( (volatile kmp_int32 *) spin )
                 : KMP_TEST_THEN_ADD4_32( (volatile kmp_int32 *) spin );

    KF_TRACE( 100, ( "__kmp_release: T#%d old spin(%p)=%d, set new spin=%d\n",
                     gtid, spin, old_spin, *spin ) );

    if ( __kmp_dflt_blocktime != KMP_MAX_BLOCKTIME ) {
        /* Only need to check sleep stuff if infinite block time not set */
        if ( old_spin & KMP_BARRIER_SLEEP_STATE ) {
 #ifndef KMP_DEBUG
            int target_gtid = target_thr->th.th_info.ds.ds_gtid;
 #endif
            /* wake up thread if needed */
            KF_TRACE( 50, ( "__kmp_release: T#%d waking up thread T#%d since sleep spin(%p) set\n",
                            gtid, target_gtid, spin ));
            __kmp_resume( target_gtid, spin );
        } else {
            KF_TRACE( 50, ( "__kmp_release: T#%d don't wake up thread T#%d since sleep spin(%p) not set\n",
                            gtid, target_gtid, spin ));
        }
    }
}

/* ------------------------------------------------------------------------ */

void
__kmp_infinite_loop( void )
{
    static int done = FALSE;

    while (! done) {
        KMP_YIELD( 1 );
    }
}

#define MAX_MESSAGE     512

void
__kmp_print_storage_map_gtid( int gtid, void *p1, void *p2, size_t size, char const *format, ...) {
    char buffer[MAX_MESSAGE];
    int node;
    va_list ap;

    va_start( ap, format);
    sprintf( buffer, "OMP storage map: %p %p%8lu %s\n", p1, p2, (unsigned long) size, format );
    __kmp_acquire_bootstrap_lock( & __kmp_stdio_lock );
    __kmp_vprintf( kmp_err, buffer, ap );
#if KMP_PRINT_DATA_PLACEMENT
    if(gtid >= 0) {
        if(p1 <= p2 && (char*)p2 - (char*)p1 == size) {
            if( __kmp_storage_map_verbose ) {
                node = __kmp_get_host_node(p1);
                if(node < 0)  /* doesn't work, so don't try this next time */
                    __kmp_storage_map_verbose = FALSE;
                else {
                    char *last;
                    int lastNode;
                    int localProc = __kmp_get_cpu_from_gtid(gtid);

                    p1 = (void *)( (size_t)p1 & ~((size_t)PAGE_SIZE - 1) );
                    p2 = (void *)( ((size_t) p2 - 1) & ~((size_t)PAGE_SIZE - 1) );
                    if(localProc >= 0)
                        __kmp_printf_no_lock("  GTID %d localNode %d\n", gtid, localProc>>1);
                    else
                        __kmp_printf_no_lock("  GTID %d\n", gtid);
# if KMP_USE_PRCTL
/* The more elaborate format is disabled for now because of the prctl hanging bug. */
                    do {
                        last = p1;
                        lastNode = node;
                        /* This loop collates adjacent pages with the same host node. */
                        do {
                            (char*)p1 += PAGE_SIZE;
                        } while(p1 <= p2 && (node = __kmp_get_host_node(p1)) == lastNode);
                        __kmp_printf_no_lock("    %p-%p memNode %d\n", last,
                                             (char*)p1 - 1, lastNode);
                    } while(p1 <= p2);
# else
                    __kmp_printf_no_lock("    %p-%p memNode %d\n", p1,
                                         (char*)p1 + (PAGE_SIZE - 1), __kmp_get_host_node(p1));
                    if(p1 < p2)  {
                        __kmp_printf_no_lock("    %p-%p memNode %d\n", p2,
                                             (char*)p2 + (PAGE_SIZE - 1), __kmp_get_host_node(p2));
                    }
# endif
                }
            }
        } else
            __kmp_printf_no_lock("  %s\n", KMP_I18N_STR( StorageMapWarning ) );
    }
#endif /* KMP_PRINT_DATA_PLACEMENT */
    __kmp_release_bootstrap_lock( & __kmp_stdio_lock );
}

void
__kmp_warn( char const * format, ... )
{
    char buffer[MAX_MESSAGE];
    va_list ap;

    if ( __kmp_generate_warnings == kmp_warnings_off ) {
        return;
    }

    va_start( ap, format );

    snprintf( buffer, sizeof(buffer) , "OMP warning: %s\n", format );
    __kmp_acquire_bootstrap_lock( & __kmp_stdio_lock );
    __kmp_vprintf( kmp_err, buffer, ap );
    __kmp_release_bootstrap_lock( & __kmp_stdio_lock );

    va_end( ap );
}

void
__kmp_abort_process()
{

    // Later threads may stall here, but that's ok because abort() will kill them.
    __kmp_acquire_bootstrap_lock( & __kmp_exit_lock );

    if ( __kmp_debug_buf ) {
        __kmp_dump_debug_buffer();
    }; // if

    if ( KMP_OS_WINDOWS ) {
        // Let other threads know of abnormal termination and prevent deadlock
        // if abort happened during library initialization or shutdown
        __kmp_global.g.g_abort = SIGABRT;

        /*
            On Windows* OS by default abort() causes pop-up error box, which stalls nightly testing.
            Unfortunately, we cannot reliably suppress pop-up error boxes. _set_abort_behavior()
            works well, but this function is not available in VS7 (this is not problem for DLL, but
            it is a problem for static OpenMP RTL). SetErrorMode (and so, timelimit utility) does
            not help, at least in some versions of MS C RTL.

            It seems following sequence is the only way to simulate abort() and avoid pop-up error
            box.
        */
        raise( SIGABRT );
        _exit( 3 );    // Just in case, if signal ignored, exit anyway.
    } else {
        abort();
    }; // if

    __kmp_infinite_loop();
    __kmp_release_bootstrap_lock( & __kmp_exit_lock );

} // __kmp_abort_process

void
__kmp_abort_thread( void )
{
    // TODO: Eliminate g_abort global variable and this function.
    // In case of abort just call abort(), it will kill all the threads.
    __kmp_infinite_loop();
} // __kmp_abort_thread

/* ------------------------------------------------------------------------ */

/*
 * Print out the storage map for the major kmp_info_t thread data structures
 * that are allocated together.
 */

static void
__kmp_print_thread_storage_map( kmp_info_t *thr, int gtid )
{
    __kmp_print_storage_map_gtid( gtid, thr, thr + 1, sizeof(kmp_info_t), "th_%d", gtid );

    __kmp_print_storage_map_gtid( gtid, &thr->th.th_info, &thr->th.th_team, sizeof(kmp_desc_t),
                             "th_%d.th_info", gtid );

    __kmp_print_storage_map_gtid( gtid, &thr->th.th_local, &thr->th.th_pri_head, sizeof(kmp_local_t),
                             "th_%d.th_local", gtid );

    __kmp_print_storage_map_gtid( gtid, &thr->th.th_bar[0], &thr->th.th_bar[bs_last_barrier],
                             sizeof(kmp_balign_t) * bs_last_barrier, "th_%d.th_bar", gtid );

    __kmp_print_storage_map_gtid( gtid, &thr->th.th_bar[bs_plain_barrier],
                             &thr->th.th_bar[bs_plain_barrier+1],
                             sizeof(kmp_balign_t), "th_%d.th_bar[plain]", gtid);

    __kmp_print_storage_map_gtid( gtid, &thr->th.th_bar[bs_forkjoin_barrier],
                             &thr->th.th_bar[bs_forkjoin_barrier+1],
                             sizeof(kmp_balign_t), "th_%d.th_bar[forkjoin]", gtid);

    #if KMP_FAST_REDUCTION_BARRIER
        __kmp_print_storage_map_gtid( gtid, &thr->th.th_bar[bs_reduction_barrier],
                             &thr->th.th_bar[bs_reduction_barrier+1],
                             sizeof(kmp_balign_t), "th_%d.th_bar[reduction]", gtid);
    #endif // KMP_FAST_REDUCTION_BARRIER
}

/*
 * Print out the storage map for the major kmp_team_t team data structures
 * that are allocated together.
 */

static void
__kmp_print_team_storage_map( const char *header, kmp_team_t *team, int team_id, int num_thr )
{
    int num_disp_buff = team->t.t_max_nproc > 1 ? KMP_MAX_DISP_BUF : 2;
    __kmp_print_storage_map_gtid( -1, team, team + 1, sizeof(kmp_team_t), "%s_%d",
                             header, team_id );

    __kmp_print_storage_map_gtid( -1, &team->t.t_bar[0], &team->t.t_bar[bs_last_barrier],
                             sizeof(kmp_balign_team_t) * bs_last_barrier, "%s_%d.t_bar", header, team_id );


    __kmp_print_storage_map_gtid( -1, &team->t.t_bar[bs_plain_barrier], &team->t.t_bar[bs_plain_barrier+1],
                             sizeof(kmp_balign_team_t), "%s_%d.t_bar[plain]", header, team_id );

    __kmp_print_storage_map_gtid( -1, &team->t.t_bar[bs_forkjoin_barrier], &team->t.t_bar[bs_forkjoin_barrier+1],
                             sizeof(kmp_balign_team_t), "%s_%d.t_bar[forkjoin]", header, team_id );

    #if KMP_FAST_REDUCTION_BARRIER
        __kmp_print_storage_map_gtid( -1, &team->t.t_bar[bs_reduction_barrier], &team->t.t_bar[bs_reduction_barrier+1],
                             sizeof(kmp_balign_team_t), "%s_%d.t_bar[reduction]", header, team_id );
    #endif // KMP_FAST_REDUCTION_BARRIER

    __kmp_print_storage_map_gtid( -1, &team->t.t_dispatch[0], &team->t.t_dispatch[num_thr],
                             sizeof(kmp_disp_t) * num_thr, "%s_%d.t_dispatch", header, team_id );

    __kmp_print_storage_map_gtid( -1, &team->t.t_threads[0], &team->t.t_threads[num_thr],
                             sizeof(kmp_info_t *) * num_thr, "%s_%d.t_threads", header, team_id );

    __kmp_print_storage_map_gtid( -1, &team->t.t_disp_buffer[0], &team->t.t_disp_buffer[num_disp_buff],
                             sizeof(dispatch_shared_info_t) * num_disp_buff, "%s_%d.t_disp_buffer",
                             header, team_id );

    /*
    __kmp_print_storage_map_gtid( -1, &team->t.t_set_nproc[0], &team->t.t_set_nproc[num_thr],
                             sizeof(int) * num_thr, "%s_%d.t_set_nproc", header, team_id );

    __kmp_print_storage_map_gtid( -1, &team->t.t_set_dynamic[0], &team->t.t_set_dynamic[num_thr],
                             sizeof(int) * num_thr, "%s_%d.t_set_dynamic", header, team_id );

    __kmp_print_storage_map_gtid( -1, &team->t.t_set_nested[0], &team->t.t_set_nested[num_thr],
                             sizeof(int) * num_thr, "%s_%d.t_set_nested", header, team_id );

    __kmp_print_storage_map_gtid( -1, &team->t.t_set_blocktime[0], &team->t.t_set_blocktime[num_thr],
                             sizeof(int) * num_thr, "%s_%d.t_set_nproc", header, team_id );

    __kmp_print_storage_map_gtid( -1, &team->t.t_set_bt_intervals[0], &team->t.t_set_bt_intervals[num_thr],
                             sizeof(int) * num_thr, "%s_%d.t_set_dynamic", header, team_id );

    __kmp_print_storage_map_gtid( -1, &team->t.t_set_bt_set[0], &team->t.t_set_bt_set[num_thr],
                             sizeof(int) * num_thr, "%s_%d.t_set_nested", header, team_id );

#if OMP_30_ENABLED
    //__kmp_print_storage_map_gtid( -1, &team->t.t_set_max_active_levels[0], &team->t.t_set_max_active_levels[num_thr],
    //                        sizeof(int) * num_thr, "%s_%d.t_set_max_active_levels", header, team_id );

    __kmp_print_storage_map_gtid( -1, &team->t.t_set_sched[0], &team->t.t_set_sched[num_thr],
                             sizeof(kmp_r_sched_t) * num_thr, "%s_%d.t_set_sched", header, team_id );
#endif // OMP_30_ENABLED
#if OMP_40_ENABLED
    __kmp_print_storage_map_gtid( -1, &team->t.t_set_proc_bind[0], &team->t.t_set_proc_bind[num_thr],
                             sizeof(kmp_proc_bind_t) * num_thr, "%s_%d.t_set_proc_bind", header, team_id );
#endif
    */

    __kmp_print_storage_map_gtid( -1, &team->t.t_taskq, &team->t.t_copypriv_data,
                             sizeof(kmp_taskq_t), "%s_%d.t_taskq", header, team_id );
}

static void __kmp_init_allocator() {}
static void __kmp_fini_allocator() {}
static void __kmp_fini_allocator_thread() {}

/* ------------------------------------------------------------------------ */

#ifdef GUIDEDLL_EXPORTS
# if KMP_OS_WINDOWS


static void
__kmp_reset_lock( kmp_bootstrap_lock_t* lck ) {
    // TODO: Change to __kmp_break_bootstrap_lock().
    __kmp_init_bootstrap_lock( lck ); // make the lock released
}

static void
__kmp_reset_locks_on_process_detach( int gtid_req ) {
    int i;
    int thread_count;

    // PROCESS_DETACH is expected to be called by a thread
    // that executes ProcessExit() or FreeLibrary().
    // OS terminates other threads (except the one calling ProcessExit or FreeLibrary).
    // So, it might be safe to access the __kmp_threads[] without taking the forkjoin_lock.
    // However, in fact, some threads can be still alive here, although being about to be terminated.
    // The threads in the array with ds_thread==0 are most suspicious.
    // Actually, it can be not safe to access the __kmp_threads[].

    // TODO: does it make sense to check __kmp_roots[] ?

    // Let's check that there are no other alive threads registered with the OMP lib.
    while( 1 ) {
        thread_count = 0;
        for( i = 0; i < __kmp_threads_capacity; ++i ) {
            if( !__kmp_threads ) continue;
            kmp_info_t* th = __kmp_threads[ i ];
            if( th == NULL ) continue;
            int gtid = th->th.th_info.ds.ds_gtid;
            if( gtid == gtid_req ) continue;
            if( gtid < 0 ) continue;
            DWORD exit_val;
            int alive = __kmp_is_thread_alive( th, &exit_val );
            if( alive ) {
            ++thread_count;
            }
        }
        if( thread_count == 0 ) break; // success
    }

    // Assume that I'm alone.

    // Now it might be probably safe to check and reset locks.
    // __kmp_forkjoin_lock and __kmp_stdio_lock are expected to be reset.
    __kmp_reset_lock( &__kmp_forkjoin_lock );
    #ifdef KMP_DEBUG
    __kmp_reset_lock( &__kmp_stdio_lock );
    #endif // KMP_DEBUG


}

BOOL WINAPI
DllMain( HINSTANCE hInstDLL, DWORD fdwReason, LPVOID lpReserved ) {
    //__kmp_acquire_bootstrap_lock( &__kmp_initz_lock );

    switch( fdwReason ) {

        case DLL_PROCESS_ATTACH:
            KA_TRACE( 10, ("DllMain: PROCESS_ATTACH\n" ));

            return TRUE;

        case DLL_PROCESS_DETACH:
            KA_TRACE( 10, ("DllMain: PROCESS_DETACH T#%d\n",
                        __kmp_gtid_get_specific() ));

            if( lpReserved != NULL )
            {
                // lpReserved is used for telling the difference:
                //  lpReserved == NULL when FreeLibrary() was called,
                //  lpReserved != NULL when the process terminates.
                // When FreeLibrary() is called, worker threads remain alive.
                // So they will release the forkjoin lock by themselves.
                // When the process terminates, worker threads disappear triggering
                // the problem of unreleased forkjoin lock as described below.

                // A worker thread can take the forkjoin lock
                // in __kmp_suspend()->__kmp_rml_decrease_load_before_sleep().
                // The problem comes up if that worker thread becomes dead
                // before it releases the forkjoin lock.
                // The forkjoin lock remains taken, while the thread
                // executing DllMain()->PROCESS_DETACH->__kmp_internal_end_library() below
                // will try to take the forkjoin lock and will always fail,
                // so that the application will never finish [normally].
                // This scenario is possible if __kmpc_end() has not been executed.
                // It looks like it's not a corner case, but common cases:
                // - the main function was compiled by an alternative compiler;
                // - the main function was compiled by icl but without /Qopenmp (application with plugins);
                // - application terminates by calling C exit(), Fortran CALL EXIT() or Fortran STOP.
                // - alive foreign thread prevented __kmpc_end from doing cleanup.

                // This is a hack to work around the problem.
                // TODO: !!! to figure out something better.
                __kmp_reset_locks_on_process_detach( __kmp_gtid_get_specific() );
            }

            __kmp_internal_end_library( __kmp_gtid_get_specific() );

            return TRUE;

        case DLL_THREAD_ATTACH:
            KA_TRACE( 10, ("DllMain: THREAD_ATTACH\n" ));

            /* if we wanted to register new siblings all the time here call
             * __kmp_get_gtid(); */
            return TRUE;

        case DLL_THREAD_DETACH:
            KA_TRACE( 10, ("DllMain: THREAD_DETACH T#%d\n",
                        __kmp_gtid_get_specific() ));

            __kmp_internal_end_thread( __kmp_gtid_get_specific() );
            return TRUE;
    }

    return TRUE;
}

# endif /* KMP_OS_WINDOWS */
#endif /* GUIDEDLL_EXPORTS */


/* ------------------------------------------------------------------------ */

/* Change the library type to "status" and return the old type */
/* called from within initialization routines where __kmp_initz_lock is held */
int
__kmp_change_library( int status )
{
    int old_status;

    old_status = __kmp_yield_init & 1;  // check whether KMP_LIBRARY=throughput (even init count)

    if (status) {
        __kmp_yield_init |= 1;  // throughput => turnaround (odd init count)
    }
    else {
        __kmp_yield_init &= ~1; // turnaround => throughput (even init count)
    }

    return old_status;  // return previous setting of whether KMP_LIBRARY=throughput
}

/* ------------------------------------------------------------------------ */
/* ------------------------------------------------------------------------ */

/* __kmp_parallel_deo --
 * Wait until it's our turn.
 */
void
__kmp_parallel_deo( int *gtid_ref, int *cid_ref, ident_t *loc_ref )
{
    int gtid = *gtid_ref;
#ifdef BUILD_PARALLEL_ORDERED
    kmp_team_t *team = __kmp_team_from_gtid( gtid );
#endif /* BUILD_PARALLEL_ORDERED */

    if( __kmp_env_consistency_check ) {
        if( __kmp_threads[gtid] -> th.th_root -> r.r_active )
            __kmp_push_sync( gtid, ct_ordered_in_parallel, loc_ref, NULL );
    }
#ifdef BUILD_PARALLEL_ORDERED
    if( !team -> t.t_serialized ) {
        kmp_uint32  spins;

        KMP_MB();
        KMP_WAIT_YIELD(&team -> t.t_ordered.dt.t_value, __kmp_tid_from_gtid( gtid ), KMP_EQ, NULL);
        KMP_MB();
    }
#endif /* BUILD_PARALLEL_ORDERED */
}

/* __kmp_parallel_dxo --
 * Signal the next task.
 */

void
__kmp_parallel_dxo( int *gtid_ref, int *cid_ref, ident_t *loc_ref )
{
    int gtid = *gtid_ref;
#ifdef BUILD_PARALLEL_ORDERED
    int tid =  __kmp_tid_from_gtid( gtid );
    kmp_team_t *team = __kmp_team_from_gtid( gtid );
#endif /* BUILD_PARALLEL_ORDERED */

    if( __kmp_env_consistency_check ) {
        if( __kmp_threads[gtid] -> th.th_root -> r.r_active )
            __kmp_pop_sync( gtid, ct_ordered_in_parallel, loc_ref );
    }
#ifdef BUILD_PARALLEL_ORDERED
    if ( ! team -> t.t_serialized ) {
        KMP_MB();       /* Flush all pending memory write invalidates.  */

        /* use the tid of the next thread in this team */
        /* TODO repleace with general release procedure */
        team -> t.t_ordered.dt.t_value = ((tid + 1) % team->t.t_nproc );

        KMP_MB();       /* Flush all pending memory write invalidates.  */
    }
#endif /* BUILD_PARALLEL_ORDERED */
}

/* ------------------------------------------------------------------------ */
/* ------------------------------------------------------------------------ */

/* ------------------------------------------------------------------------ */
/* ------------------------------------------------------------------------ */

/* The BARRIER for a SINGLE process section is always explicit   */

int
__kmp_enter_single( int gtid, ident_t *id_ref, int push_ws )
{
    int status;
    kmp_info_t *th;
    kmp_team_t *team;

    if( ! TCR_4(__kmp_init_parallel) )
        __kmp_parallel_initialize();

    th   = __kmp_threads[ gtid ];
    team = th -> th.th_team;
    status = 0;

    th->th.th_ident = id_ref;

    if ( team -> t.t_serialized ) {
        status = 1;
    } else {
        kmp_int32 old_this = th->th.th_local.this_construct;

        ++th->th.th_local.this_construct;
        /* try to set team count to thread count--success means thread got the
           single block
        */
        /* TODO: Should this be acquire or release? */
        status = KMP_COMPARE_AND_STORE_ACQ32(&team -> t.t_construct, old_this,
                                             th->th.th_local.this_construct);
    }

    if( __kmp_env_consistency_check ) {
        if (status && push_ws) {
            __kmp_push_workshare( gtid, ct_psingle, id_ref );
        } else {
            __kmp_check_workshare( gtid, ct_psingle, id_ref );
        }
    }
#if USE_ITT_BUILD
    if ( status ) {
        __kmp_itt_single_start( gtid );
    }
#endif /* USE_ITT_BUILD */
    return status;
}

void
__kmp_exit_single( int gtid )
{
#if USE_ITT_BUILD
    __kmp_itt_single_end( gtid );
#endif /* USE_ITT_BUILD */
    if( __kmp_env_consistency_check )
        __kmp_pop_workshare( gtid, ct_psingle, NULL );
}


/* ------------------------------------------------------------------------ */
/* ------------------------------------------------------------------------ */

static void
__kmp_linear_barrier_gather( enum barrier_type bt,
                             kmp_info_t *this_thr,
                             int gtid,
                             int tid,
                             void (*reduce)(void *, void *)
                             USE_ITT_BUILD_ARG(void * itt_sync_obj)
                             )
{
    register kmp_team_t    *team          = this_thr -> th.th_team;
    register kmp_bstate_t  *thr_bar       = & this_thr -> th.th_bar[ bt ].bb;
    register kmp_info_t   **other_threads = team -> t.t_threads;

    KA_TRACE( 20, ("__kmp_linear_barrier_gather: T#%d(%d:%d) enter for barrier type %d\n",
                   gtid, team->t.t_id, tid, bt ) );

    KMP_DEBUG_ASSERT( this_thr == other_threads[this_thr->th.th_info.ds.ds_tid] );

    /*
     * We now perform a linear reduction to signal that all
     * of the threads have arrived.
     *
     * Collect all the worker team member threads.
     */
    if ( ! KMP_MASTER_TID( tid )) {

        KA_TRACE( 20, ( "__kmp_linear_barrier_gather: T#%d(%d:%d) releasing T#%d(%d:%d)"
                        "arrived(%p): %u => %u\n",
                        gtid, team->t.t_id, tid,
                        __kmp_gtid_from_tid( 0, team ), team->t.t_id, 0,
                        &thr_bar -> b_arrived, thr_bar -> b_arrived,
                        thr_bar -> b_arrived + KMP_BARRIER_STATE_BUMP
                      ) );

        /* mark arrival to master thread */
        //
        // After performing this write, a worker thread may not assume that
        // the team is valid any more - it could be deallocated by the master
        // thread at any time.
        //
        __kmp_release( other_threads[0], &thr_bar -> b_arrived, kmp_release_fence );

    } else {
        register kmp_balign_team_t *team_bar  = & team -> t.t_bar[ bt ];
        register int                nproc     = this_thr -> th.th_team_nproc;
        register int                i;
        /* Don't have to worry about sleep bit here or atomic since team setting */
        register kmp_uint           new_state  = team_bar -> b_arrived + KMP_BARRIER_STATE_BUMP;

        /* Collect all the worker team member threads. */
        for (i = 1; i < nproc; i++) {
#if KMP_CACHE_MANAGE
            /* prefetch next thread's arrived count */
            if ( i+1 < nproc )
                KMP_CACHE_PREFETCH( &other_threads[ i+1 ] -> th.th_bar[ bt ].bb.b_arrived );
#endif /* KMP_CACHE_MANAGE */
            KA_TRACE( 20, ( "__kmp_linear_barrier_gather: T#%d(%d:%d) wait T#%d(%d:%d) "
                            "arrived(%p) == %u\n",
                            gtid, team->t.t_id, tid,
                            __kmp_gtid_from_tid( i, team ), team->t.t_id, i,
                            &other_threads[i] -> th.th_bar[ bt ].bb.b_arrived,
                            new_state ) );

            /* wait for worker thread to arrive */
            __kmp_wait_sleep( this_thr,
                              & other_threads[ i ] -> th.th_bar[ bt ].bb.b_arrived,
                              new_state, FALSE
                              USE_ITT_BUILD_ARG( itt_sync_obj )
                              );

            if (reduce) {

                KA_TRACE( 100, ( "__kmp_linear_barrier_gather: T#%d(%d:%d) += T#%d(%d:%d)\n",
                                 gtid, team->t.t_id, tid,
                                 __kmp_gtid_from_tid( i, team ), team->t.t_id, i ) );

                (*reduce)( this_thr -> th.th_local.reduce_data,
                           other_threads[ i ] -> th.th_local.reduce_data );

            }

        }

        /* Don't have to worry about sleep bit here or atomic since team setting */
        team_bar -> b_arrived = new_state;
        KA_TRACE( 20, ( "__kmp_linear_barrier_gather: T#%d(%d:%d) set team %d "
                        "arrived(%p) = %u\n",
                        gtid, team->t.t_id, tid, team->t.t_id,
                        &team_bar -> b_arrived, new_state ) );
    }

    KA_TRACE( 20, ( "__kmp_linear_barrier_gather: T#%d(%d:%d) exit for barrier type %d\n",
                    gtid, team->t.t_id, tid, bt ) );
}


static void
__kmp_tree_barrier_gather( enum barrier_type bt,
                           kmp_info_t *this_thr,
                           int gtid,
                           int tid,
                           void (*reduce) (void *, void *)
                           USE_ITT_BUILD_ARG( void * itt_sync_obj )
                           )
{
    register kmp_team_t    *team          = this_thr -> th.th_team;
    register kmp_bstate_t  *thr_bar       = & this_thr -> th.th_bar[ bt ].bb;
    register kmp_info_t   **other_threads = team -> t.t_threads;
    register kmp_uint32     nproc         = this_thr -> th.th_team_nproc;
    register kmp_uint32     branch_bits   = __kmp_barrier_gather_branch_bits[ bt ];
    register kmp_uint32     branch_factor = 1 << branch_bits ;
    register kmp_uint32     child;
    register kmp_uint32     child_tid;
    register kmp_uint       new_state;

    KA_TRACE( 20, ( "__kmp_tree_barrier_gather: T#%d(%d:%d) enter for barrier type %d\n",
                    gtid, team->t.t_id, tid, bt ) );

    KMP_DEBUG_ASSERT( this_thr == other_threads[this_thr->th.th_info.ds.ds_tid] );

    /*
     * We now perform a tree gather to wait until all
     * of the threads have arrived, and reduce any required data
     * as we go.
     */

    child_tid = (tid << branch_bits) + 1;

    if ( child_tid < nproc ) {

        /* parent threads wait for all their children to arrive */
        new_state = team -> t.t_bar[ bt ].b_arrived + KMP_BARRIER_STATE_BUMP;
        child = 1;

        do {
            register kmp_info_t   *child_thr = other_threads[ child_tid ];
            register kmp_bstate_t *child_bar = & child_thr -> th.th_bar[ bt ].bb;
#if KMP_CACHE_MANAGE
            /* prefetch next thread's arrived count */
            if ( child+1 <= branch_factor && child_tid+1 < nproc )
                KMP_CACHE_PREFETCH( &other_threads[ child_tid+1 ] -> th.th_bar[ bt ].bb.b_arrived );
#endif /* KMP_CACHE_MANAGE */
            KA_TRACE( 20, ( "__kmp_tree_barrier_gather: T#%d(%d:%d) wait T#%d(%d:%u) "
                            "arrived(%p) == %u\n",
                            gtid, team->t.t_id, tid,
                            __kmp_gtid_from_tid( child_tid, team ), team->t.t_id, child_tid,
                            &child_bar -> b_arrived, new_state ) );

            /* wait for child to arrive */
            __kmp_wait_sleep( this_thr, &child_bar -> b_arrived, new_state, FALSE
                              USE_ITT_BUILD_ARG( itt_sync_obj)
                              );

            if (reduce) {

                KA_TRACE( 100, ( "__kmp_tree_barrier_gather: T#%d(%d:%d) += T#%d(%d:%u)\n",
                                 gtid, team->t.t_id, tid,
                                 __kmp_gtid_from_tid( child_tid, team ), team->t.t_id,
                                 child_tid ) );

                (*reduce)( this_thr -> th.th_local.reduce_data,
                           child_thr -> th.th_local.reduce_data );

            }

            child++;
            child_tid++;
        }
        while ( child <= branch_factor && child_tid < nproc );
    }

    if ( !KMP_MASTER_TID(tid) ) {
        /* worker threads */
        register kmp_int32 parent_tid = (tid - 1) >> branch_bits;

        KA_TRACE( 20, ( "__kmp_tree_barrier_gather: T#%d(%d:%d) releasing T#%d(%d:%d) "
                        "arrived(%p): %u => %u\n",
                        gtid, team->t.t_id, tid,
                        __kmp_gtid_from_tid( parent_tid, team ), team->t.t_id, parent_tid,
                        &thr_bar -> b_arrived, thr_bar -> b_arrived,
                        thr_bar -> b_arrived + KMP_BARRIER_STATE_BUMP
                      ) );

        /* mark arrival to parent thread */
        //
        // After performing this write, a worker thread may not assume that
        // the team is valid any more - it could be deallocated by the master
        // thread at any time.
        //
        __kmp_release( other_threads[parent_tid], &thr_bar -> b_arrived, kmp_release_fence );

    } else {
        /* Need to update the team arrived pointer if we are the master thread */

        if ( nproc > 1 )
            /* New value was already computed above */
            team -> t.t_bar[ bt ].b_arrived = new_state;
        else
            team -> t.t_bar[ bt ].b_arrived += KMP_BARRIER_STATE_BUMP;

        KA_TRACE( 20, ( "__kmp_tree_barrier_gather: T#%d(%d:%d) set team %d arrived(%p) = %u\n",
                        gtid, team->t.t_id, tid, team->t.t_id,
                        &team->t.t_bar[bt].b_arrived, team->t.t_bar[bt].b_arrived ) );
    }

    KA_TRACE( 20, ( "__kmp_tree_barrier_gather: T#%d(%d:%d) exit for barrier type %d\n",
                    gtid, team->t.t_id, tid, bt ) );
}


static void
__kmp_hyper_barrier_gather( enum barrier_type bt,
                            kmp_info_t *this_thr,
                            int gtid,
                            int tid,
                            void (*reduce) (void *, void *)
                            USE_ITT_BUILD_ARG (void * itt_sync_obj)
                            )
{
    register kmp_team_t    *team          = this_thr -> th.th_team;
    register kmp_bstate_t  *thr_bar       = & this_thr -> th.th_bar[ bt ].bb;
    register kmp_info_t   **other_threads = team -> t.t_threads;
    register kmp_uint       new_state     = KMP_BARRIER_UNUSED_STATE;
    register kmp_uint32     num_threads   = this_thr -> th.th_team_nproc;
    register kmp_uint32     branch_bits   = __kmp_barrier_gather_branch_bits[ bt ];
    register kmp_uint32     branch_factor = 1 << branch_bits ;
    register kmp_uint32     offset;
    register kmp_uint32     level;

    KA_TRACE( 20, ( "__kmp_hyper_barrier_gather: T#%d(%d:%d) enter for barrier type %d\n",
                    gtid, team->t.t_id, tid, bt ) );

    KMP_DEBUG_ASSERT( this_thr == other_threads[this_thr->th.th_info.ds.ds_tid] );

#if USE_ITT_BUILD && USE_ITT_NOTIFY
    // Barrier imbalance - save arrive time to the thread
    if( __kmp_forkjoin_frames_mode == 2 || __kmp_forkjoin_frames_mode == 3 ) {
        this_thr->th.th_bar_arrive_time = __itt_get_timestamp();
    }
#endif
    /*
     * We now perform a hypercube-embedded tree gather to wait until all
     * of the threads have arrived, and reduce any required data
     * as we go.
     */

    for ( level=0, offset =1;
          offset < num_threads;
          level += branch_bits, offset <<= branch_bits )
    {
        register kmp_uint32     child;
        register kmp_uint32 child_tid;

        if ( ((tid >> level) & (branch_factor - 1)) != 0 ) {
            register kmp_int32 parent_tid = tid & ~( (1 << (level + branch_bits)) -1 );

            KA_TRACE( 20, ( "__kmp_hyper_barrier_gather: T#%d(%d:%d) releasing T#%d(%d:%d) "
                            "arrived(%p): %u => %u\n",
                            gtid, team->t.t_id, tid,
                            __kmp_gtid_from_tid( parent_tid, team ), team->t.t_id, parent_tid,
                            &thr_bar -> b_arrived, thr_bar -> b_arrived,
                            thr_bar -> b_arrived + KMP_BARRIER_STATE_BUMP
                          ) );

            /* mark arrival to parent thread */
            //
            // After performing this write (in the last iteration of the
            // enclosing for loop), a worker thread may not assume that the
            // team is valid any more - it could be deallocated by the master
            // thread at any time.
            //
            __kmp_release( other_threads[parent_tid], &thr_bar -> b_arrived, kmp_release_fence );
            break;
        }

        /* parent threads wait for children to arrive */

        if (new_state == KMP_BARRIER_UNUSED_STATE)
            new_state = team -> t.t_bar[ bt ].b_arrived + KMP_BARRIER_STATE_BUMP;

        for ( child = 1, child_tid = tid + (1 << level);
              child < branch_factor && child_tid < num_threads;
              child++, child_tid += (1 << level) )
        {
            register kmp_info_t   *child_thr = other_threads[ child_tid ];
            register kmp_bstate_t *child_bar = & child_thr -> th.th_bar[ bt ].bb;
#if KMP_CACHE_MANAGE
            register kmp_uint32 next_child_tid = child_tid + (1 << level);
            /* prefetch next thread's arrived count */
            if ( child+1 < branch_factor && next_child_tid < num_threads )
                KMP_CACHE_PREFETCH( &other_threads[ next_child_tid ] -> th.th_bar[ bt ].bb.b_arrived );
#endif /* KMP_CACHE_MANAGE */
            KA_TRACE( 20, ( "__kmp_hyper_barrier_gather: T#%d(%d:%d) wait T#%d(%d:%u) "
                            "arrived(%p) == %u\n",
                            gtid, team->t.t_id, tid,
                            __kmp_gtid_from_tid( child_tid, team ), team->t.t_id, child_tid,
                            &child_bar -> b_arrived, new_state ) );

            /* wait for child to arrive */
            __kmp_wait_sleep( this_thr, &child_bar -> b_arrived, new_state, FALSE
                              USE_ITT_BUILD_ARG (itt_sync_obj)
                              );

#if USE_ITT_BUILD
            // Barrier imbalance - write min of the thread time and a child time to the thread.
            if( __kmp_forkjoin_frames_mode == 2 || __kmp_forkjoin_frames_mode == 3 ) {
                this_thr->th.th_bar_arrive_time = KMP_MIN( this_thr->th.th_bar_arrive_time, child_thr->th.th_bar_arrive_time );
            }
#endif
            if (reduce) {

                KA_TRACE( 100, ( "__kmp_hyper_barrier_gather: T#%d(%d:%d) += T#%d(%d:%u)\n",
                                 gtid, team->t.t_id, tid,
                                 __kmp_gtid_from_tid( child_tid, team ), team->t.t_id,
                                 child_tid ) );

                (*reduce)( this_thr -> th.th_local.reduce_data,
                           child_thr -> th.th_local.reduce_data );

            }
        }
    }


    if ( KMP_MASTER_TID(tid) ) {
        /* Need to update the team arrived pointer if we are the master thread */

        if (new_state == KMP_BARRIER_UNUSED_STATE)
            team -> t.t_bar[ bt ].b_arrived += KMP_BARRIER_STATE_BUMP;
        else
            team -> t.t_bar[ bt ].b_arrived = new_state;

        KA_TRACE( 20, ( "__kmp_hyper_barrier_gather: T#%d(%d:%d) set team %d arrived(%p) = %u\n",
                        gtid, team->t.t_id, tid, team->t.t_id,
                        &team->t.t_bar[bt].b_arrived, team->t.t_bar[bt].b_arrived ) );
    }

    KA_TRACE( 20, ( "__kmp_hyper_barrier_gather: T#%d(%d:%d) exit for barrier type %d\n",
                    gtid, team->t.t_id, tid, bt ) );

}

static void
__kmp_linear_barrier_release( enum barrier_type bt,
                              kmp_info_t *this_thr,
                              int gtid,
                              int tid,
                              int propagate_icvs
                              USE_ITT_BUILD_ARG(void * itt_sync_obj)
                              )
{
    register kmp_bstate_t *thr_bar = &this_thr -> th.th_bar[ bt ].bb;
    register kmp_team_t *team;

    if (KMP_MASTER_TID( tid )) {
        register unsigned int i;
        register kmp_uint32 nproc = this_thr -> th.th_team_nproc;
        register kmp_info_t **other_threads;

        team = __kmp_threads[ gtid ]-> th.th_team;
        KMP_DEBUG_ASSERT( team != NULL );
        other_threads = team -> t.t_threads;

        KA_TRACE( 20, ( "__kmp_linear_barrier_release: T#%d(%d:%d) master enter for barrier type %d\n",
          gtid, team->t.t_id, tid, bt ) );

        if (nproc > 1) {
#if KMP_BARRIER_ICV_PUSH
            if ( propagate_icvs ) {
                load_icvs(&team->t.t_implicit_task_taskdata[0].td_icvs);
                for (i = 1; i < nproc; i++) {
                    __kmp_init_implicit_task( team->t.t_ident,
                                              team->t.t_threads[i], team, i, FALSE );
                    store_icvs(&team->t.t_implicit_task_taskdata[i].td_icvs, &team->t.t_implicit_task_taskdata[0].td_icvs);
                }
                sync_icvs();
            }
#endif // KMP_BARRIER_ICV_PUSH

            /* Now, release all of the worker threads */
            for (i = 1; i < nproc; i++) {
#if KMP_CACHE_MANAGE
                /* prefetch next thread's go flag */
                if( i+1 < nproc )
                    KMP_CACHE_PREFETCH( &other_threads[ i+1 ]-> th.th_bar[ bt ].bb.b_go );
#endif /* KMP_CACHE_MANAGE */
                KA_TRACE( 20, ( "__kmp_linear_barrier_release: T#%d(%d:%d) releasing T#%d(%d:%d) "
                                "go(%p): %u => %u\n",
                                gtid, team->t.t_id, tid,
                                other_threads[i]->th.th_info.ds.ds_gtid, team->t.t_id, i,
                                &other_threads[i]->th.th_bar[bt].bb.b_go,
                                other_threads[i]->th.th_bar[bt].bb.b_go,
                                other_threads[i]->th.th_bar[bt].bb.b_go + KMP_BARRIER_STATE_BUMP
                                ) );

                __kmp_release( other_threads[ i ],
                               &other_threads[ i ]-> th.th_bar[ bt ].bb.b_go, kmp_acquire_fence );
            }
        }
    } else {
        /* Wait for the MASTER thread to release us */

        KA_TRACE( 20, ( "__kmp_linear_barrier_release: T#%d wait go(%p) == %u\n",
          gtid, &thr_bar -> b_go, KMP_BARRIER_STATE_BUMP ) );

        __kmp_wait_sleep( this_thr, &thr_bar -> b_go, KMP_BARRIER_STATE_BUMP, TRUE
                          USE_ITT_BUILD_ARG(itt_sync_obj)
                          );

#if USE_ITT_BUILD && OMP_30_ENABLED && USE_ITT_NOTIFY
        if ( ( __itt_sync_create_ptr && itt_sync_obj == NULL ) || KMP_ITT_DEBUG ) {
            // we are on a fork barrier where we could not get the object reliably (or ITTNOTIFY is disabled)
            itt_sync_obj  = __kmp_itt_barrier_object( gtid, bs_forkjoin_barrier, 0, -1 );
            // cancel wait on previous parallel region...
            __kmp_itt_task_starting( itt_sync_obj );

            if ( bt == bs_forkjoin_barrier && TCR_4(__kmp_global.g.g_done) )
                return;

            itt_sync_obj  = __kmp_itt_barrier_object( gtid, bs_forkjoin_barrier );
            if ( itt_sync_obj != NULL )
                __kmp_itt_task_finished( itt_sync_obj );  // call prepare as early as possible for "new" barrier

        } else
#endif /* USE_ITT_BUILD && OMP_30_ENABLED && USE_ITT_NOTIFY */
        //
        // early exit for reaping threads releasing forkjoin barrier
        //
        if ( bt == bs_forkjoin_barrier && TCR_4(__kmp_global.g.g_done) )
            return;

        //
        // The worker thread may now assume that the team is valid.
        //
#if USE_ITT_BUILD && !OMP_30_ENABLED && USE_ITT_NOTIFY
        // libguide only code (cannot use *itt_task* routines)
        if ( ( __itt_sync_create_ptr && itt_sync_obj == NULL ) || KMP_ITT_DEBUG ) {
            // we are on a fork barrier where we could not get the object reliably
            itt_sync_obj  = __kmp_itt_barrier_object( gtid, bs_forkjoin_barrier );
            __kmp_itt_barrier_starting( gtid, itt_sync_obj );  // no need to call releasing, but we have paired calls...
        }
#endif /* USE_ITT_BUILD && !OMP_30_ENABLED && USE_ITT_NOTIFY */
        #ifdef KMP_DEBUG
            tid = __kmp_tid_from_gtid( gtid );
            team = __kmp_threads[ gtid ]-> th.th_team;
        #endif
        KMP_DEBUG_ASSERT( team != NULL );

        TCW_4(thr_bar->b_go, KMP_INIT_BARRIER_STATE);
        KA_TRACE( 20, ("__kmp_linear_barrier_release: T#%d(%d:%d) set go(%p) = %u\n",
          gtid, team->t.t_id, tid, &thr_bar->b_go, KMP_INIT_BARRIER_STATE ) );

        KMP_MB();       /* Flush all pending memory write invalidates.  */
    }

    KA_TRACE( 20, ( "__kmp_linear_barrier_release: T#%d(%d:%d) exit for barrier type %d\n",
      gtid, team->t.t_id, tid, bt ) );
}


static void
__kmp_tree_barrier_release( enum barrier_type bt,
                            kmp_info_t *this_thr,
                            int gtid,
                            int tid,
                            int propagate_icvs
                            USE_ITT_BUILD_ARG(void * itt_sync_obj)
                            )
{
    /* handle fork barrier workers who aren't part of a team yet */
    register kmp_team_t    *team;
    register kmp_bstate_t  *thr_bar       = & this_thr -> th.th_bar[ bt ].bb;
    register kmp_uint32     nproc;
    register kmp_uint32     branch_bits   = __kmp_barrier_release_branch_bits[ bt ];
    register kmp_uint32     branch_factor = 1 << branch_bits ;
    register kmp_uint32     child;
    register kmp_uint32     child_tid;

    /*
     * We now perform a tree release for all
     * of the threads that have been gathered
     */

    if ( ! KMP_MASTER_TID( tid )) {
        /* worker threads */

        KA_TRACE( 20, ( "__kmp_tree_barrier_release: T#%d wait go(%p) == %u\n",
          gtid, &thr_bar -> b_go, KMP_BARRIER_STATE_BUMP ) );

        /* wait for parent thread to release us */
        __kmp_wait_sleep( this_thr, &thr_bar -> b_go, KMP_BARRIER_STATE_BUMP, TRUE
                          USE_ITT_BUILD_ARG(itt_sync_obj)
                          );

#if USE_ITT_BUILD && OMP_30_ENABLED && USE_ITT_NOTIFY
        if ( ( __itt_sync_create_ptr && itt_sync_obj == NULL ) || KMP_ITT_DEBUG ) {
            // we are on a fork barrier where we could not get the object reliably (or ITTNOTIFY is disabled)
            itt_sync_obj  = __kmp_itt_barrier_object( gtid, bs_forkjoin_barrier, 0, -1 );
            // cancel wait on previous parallel region...
            __kmp_itt_task_starting( itt_sync_obj );

            if ( bt == bs_forkjoin_barrier && TCR_4(__kmp_global.g.g_done) )
                return;

            itt_sync_obj  = __kmp_itt_barrier_object( gtid, bs_forkjoin_barrier );
            if ( itt_sync_obj != NULL )
                __kmp_itt_task_finished( itt_sync_obj );  // call prepare as early as possible for "new" barrier

        } else
#endif /* USE_ITT_BUILD && OMP_30_ENABLED && USE_ITT_NOTIFY */
        //
        // early exit for reaping threads releasing forkjoin barrier
        //
        if ( bt == bs_forkjoin_barrier && TCR_4(__kmp_global.g.g_done) )
            return;

        //
        // The worker thread may now assume that the team is valid.
        //
#if USE_ITT_BUILD && !OMP_30_ENABLED && USE_ITT_NOTIFY
        // libguide only code (cannot use *itt_task* routines)
        if ( ( __itt_sync_create_ptr && itt_sync_obj == NULL ) || KMP_ITT_DEBUG ) {
            // we are on a fork barrier where we could not get the object reliably
            itt_sync_obj  = __kmp_itt_barrier_object( gtid, bs_forkjoin_barrier );
            __kmp_itt_barrier_starting( gtid, itt_sync_obj );  // no need to call releasing, but we have paired calls...
        }
#endif /* USE_ITT_BUILD && !OMP_30_ENABLED && USE_ITT_NOTIFY */
        team = __kmp_threads[ gtid ]-> th.th_team;
        KMP_DEBUG_ASSERT( team != NULL );
        tid = __kmp_tid_from_gtid( gtid );

        TCW_4(thr_bar->b_go, KMP_INIT_BARRIER_STATE);
        KA_TRACE( 20, ( "__kmp_tree_barrier_release: T#%d(%d:%d) set go(%p) = %u\n",
          gtid, team->t.t_id, tid, &thr_bar->b_go, KMP_INIT_BARRIER_STATE ) );

        KMP_MB();       /* Flush all pending memory write invalidates.  */

    } else {
        team = __kmp_threads[ gtid ]-> th.th_team;
        KMP_DEBUG_ASSERT( team != NULL );

        KA_TRACE( 20, ( "__kmp_tree_barrier_release: T#%d(%d:%d) master enter for barrier type %d\n",
          gtid, team->t.t_id, tid, bt ) );
    }

    nproc     = this_thr -> th.th_team_nproc;
    child_tid = ( tid << branch_bits ) + 1;

    if ( child_tid < nproc ) {
        register kmp_info_t **other_threads = team -> t.t_threads;
        child = 1;
        /* parent threads release all their children */

        do {
            register kmp_info_t   *child_thr = other_threads[ child_tid ];
            register kmp_bstate_t *child_bar = & child_thr -> th.th_bar[ bt ].bb;
#if KMP_CACHE_MANAGE
            /* prefetch next thread's go count */
            if ( child+1 <= branch_factor && child_tid+1 < nproc )
                KMP_CACHE_PREFETCH( &other_threads[ child_tid+1 ] -> th.th_bar[ bt ].bb.b_go );
#endif /* KMP_CACHE_MANAGE */

#if KMP_BARRIER_ICV_PUSH
            if ( propagate_icvs ) {
                __kmp_init_implicit_task( team->t.t_ident,
                  team->t.t_threads[child_tid], team, child_tid, FALSE );
                load_icvs(&team->t.t_implicit_task_taskdata[0].td_icvs);
                store_icvs(&team->t.t_implicit_task_taskdata[child_tid].td_icvs, &team->t.t_implicit_task_taskdata[0].td_icvs);
                sync_icvs();
            }
#endif // KMP_BARRIER_ICV_PUSH

            KA_TRACE( 20, ( "__kmp_tree_barrier_release: T#%d(%d:%d) releasing T#%d(%d:%u)"
                            "go(%p): %u => %u\n",
                            gtid, team->t.t_id, tid,
                            __kmp_gtid_from_tid( child_tid, team ), team->t.t_id,
                            child_tid, &child_bar -> b_go, child_bar -> b_go,
                            child_bar -> b_go + KMP_BARRIER_STATE_BUMP ) );

            /* release child from barrier */
            __kmp_release( child_thr, &child_bar -> b_go, kmp_acquire_fence );

            child++;
            child_tid++;
        }
        while ( child <= branch_factor && child_tid < nproc );
    }

    KA_TRACE( 20, ( "__kmp_tree_barrier_release: T#%d(%d:%d) exit for barrier type %d\n",
      gtid, team->t.t_id, tid, bt ) );
}

/* The reverse versions seem to beat the forward versions overall */
#define KMP_REVERSE_HYPER_BAR
static void
__kmp_hyper_barrier_release( enum barrier_type bt,
                             kmp_info_t *this_thr,
                             int gtid,
                             int tid,
                             int propagate_icvs
                             USE_ITT_BUILD_ARG(void * itt_sync_obj)
                             )
{
    /* handle fork barrier workers who aren't part of a team yet */
    register kmp_team_t    *team;
    register kmp_bstate_t  *thr_bar       = & this_thr -> th.th_bar[ bt ].bb;
    register kmp_info_t   **other_threads;
    register kmp_uint32     num_threads;
    register kmp_uint32     branch_bits   = __kmp_barrier_release_branch_bits[ bt ];
    register kmp_uint32     branch_factor = 1 << branch_bits;
    register kmp_uint32     child;
    register kmp_uint32     child_tid;
    register kmp_uint32     offset;
    register kmp_uint32     level;

    /* Perform a hypercube-embedded tree release for all of the threads
       that have been gathered.  If KMP_REVERSE_HYPER_BAR is defined (default)
       the threads are released in the reverse order of the corresponding gather,
       otherwise threads are released in the same order. */

    if ( ! KMP_MASTER_TID( tid )) {
        /* worker threads */
        KA_TRACE( 20, ( "__kmp_hyper_barrier_release: T#%d wait go(%p) == %u\n",
          gtid, &thr_bar -> b_go, KMP_BARRIER_STATE_BUMP ) );

        /* wait for parent thread to release us */
        __kmp_wait_sleep( this_thr, &thr_bar -> b_go, KMP_BARRIER_STATE_BUMP, TRUE
                          USE_ITT_BUILD_ARG( itt_sync_obj )
                          );

#if USE_ITT_BUILD && OMP_30_ENABLED && USE_ITT_NOTIFY
        if ( ( __itt_sync_create_ptr && itt_sync_obj == NULL ) || KMP_ITT_DEBUG ) {
            // we are on a fork barrier where we could not get the object reliably
            itt_sync_obj  = __kmp_itt_barrier_object( gtid, bs_forkjoin_barrier, 0, -1 );
            // cancel wait on previous parallel region...
            __kmp_itt_task_starting( itt_sync_obj );

            if ( bt == bs_forkjoin_barrier && TCR_4(__kmp_global.g.g_done) )
                return;

            itt_sync_obj  = __kmp_itt_barrier_object( gtid, bs_forkjoin_barrier );
            if ( itt_sync_obj != NULL )
                __kmp_itt_task_finished( itt_sync_obj );  // call prepare as early as possible for "new" barrier

        } else
#endif /* USE_ITT_BUILD && OMP_30_ENABLED && USE_ITT_NOTIFY */
        //
        // early exit for reaping threads releasing forkjoin barrier
        //
        if ( bt == bs_forkjoin_barrier && TCR_4(__kmp_global.g.g_done) )
            return;

        //
        // The worker thread may now assume that the team is valid.
        //
#if USE_ITT_BUILD && !OMP_30_ENABLED && USE_ITT_NOTIFY
        // libguide only code (cannot use *itt_task* routines)
        if ( ( __itt_sync_create_ptr && itt_sync_obj == NULL ) || KMP_ITT_DEBUG ) {
            // we are on a fork barrier where we could not get the object reliably
            itt_sync_obj  = __kmp_itt_barrier_object( gtid, bs_forkjoin_barrier );
            __kmp_itt_barrier_starting( gtid, itt_sync_obj );  // no need to call releasing, but we have paired calls...
        }
#endif /* USE_ITT_BUILD && !OMP_30_ENABLED && USE_ITT_NOTIFY */
        team = __kmp_threads[ gtid ]-> th.th_team;
        KMP_DEBUG_ASSERT( team != NULL );
        tid = __kmp_tid_from_gtid( gtid );

        TCW_4(thr_bar->b_go, KMP_INIT_BARRIER_STATE);
        KA_TRACE( 20, ( "__kmp_hyper_barrier_release: T#%d(%d:%d) set go(%p) = %u\n",
                        gtid, team->t.t_id, tid, &thr_bar->b_go, KMP_INIT_BARRIER_STATE ) );

        KMP_MB();       /* Flush all pending memory write invalidates.  */

    } else {  /* KMP_MASTER_TID(tid) */
        team = __kmp_threads[ gtid ]-> th.th_team;
        KMP_DEBUG_ASSERT( team != NULL );

        KA_TRACE( 20, ( "__kmp_hyper_barrier_release: T#%d(%d:%d) master enter for barrier type %d\n",
          gtid, team->t.t_id, tid, bt ) );
    }

    num_threads = this_thr -> th.th_team_nproc;
    other_threads = team -> t.t_threads;

#ifdef KMP_REVERSE_HYPER_BAR
    /* count up to correct level for parent */
    for ( level = 0, offset = 1;
          offset < num_threads && (((tid >> level) & (branch_factor-1)) == 0);
          level += branch_bits, offset <<= branch_bits );

    /* now go down from there */
    for ( level -= branch_bits, offset >>= branch_bits;
          offset != 0;
          level -= branch_bits, offset >>= branch_bits )
#else
    /* Go down the tree, level by level */
    for ( level = 0, offset = 1;
          offset < num_threads;
          level += branch_bits, offset <<= branch_bits )
#endif // KMP_REVERSE_HYPER_BAR
    {
#ifdef KMP_REVERSE_HYPER_BAR
        /* Now go in reverse order through the children, highest to lowest.
           Initial setting of child is conservative here. */
        child = num_threads >> ((level==0)?level:level-1);
        for ( child = (child < branch_factor-1) ? child : branch_factor-1,
                  child_tid = tid + (child << level);
              child >= 1;
              child--, child_tid -= (1 << level) )
#else
        if (((tid >> level) & (branch_factor - 1)) != 0)
            /* No need to go any lower than this, since this is the level
               parent would be notified */
            break;

        /* iterate through children on this level of the tree */
        for ( child = 1, child_tid = tid + (1 << level);
              child < branch_factor && child_tid < num_threads;
              child++, child_tid += (1 << level) )
#endif // KMP_REVERSE_HYPER_BAR
        {
            if ( child_tid >= num_threads ) continue;   /* child doesn't exist so keep going */
            else {
                register kmp_info_t   *child_thr = other_threads[ child_tid ];
                register kmp_bstate_t *child_bar = & child_thr -> th.th_bar[ bt ].bb;
#if KMP_CACHE_MANAGE
                register kmp_uint32 next_child_tid = child_tid - (1 << level);
                /* prefetch next thread's go count */
#ifdef KMP_REVERSE_HYPER_BAR
                if ( child-1 >= 1 && next_child_tid < num_threads )
#else
                if ( child+1 < branch_factor && next_child_tid < num_threads )
#endif // KMP_REVERSE_HYPER_BAR
                    KMP_CACHE_PREFETCH( &other_threads[ next_child_tid ]->th.th_bar[ bt ].bb.b_go );
#endif /* KMP_CACHE_MANAGE */

#if KMP_BARRIER_ICV_PUSH
                if ( propagate_icvs ) {
                    KMP_DEBUG_ASSERT( team != NULL );
                    __kmp_init_implicit_task( team->t.t_ident,
                      team->t.t_threads[child_tid], team, child_tid, FALSE );
                    load_icvs(&team->t.t_implicit_task_taskdata[0].td_icvs);
                    store_icvs(&team->t.t_implicit_task_taskdata[child_tid].td_icvs, &team->t.t_implicit_task_taskdata[0].td_icvs);
                    sync_icvs();
                }
#endif // KMP_BARRIER_ICV_PUSH

                KA_TRACE( 20, ( "__kmp_hyper_barrier_release: T#%d(%d:%d) releasing T#%d(%d:%u)"
                                "go(%p): %u => %u\n",
                                gtid, team->t.t_id, tid,
                                __kmp_gtid_from_tid( child_tid, team ), team->t.t_id,
                                child_tid, &child_bar -> b_go, child_bar -> b_go,
                                child_bar -> b_go + KMP_BARRIER_STATE_BUMP ) );

                /* release child from barrier */
                __kmp_release( child_thr, &child_bar -> b_go, kmp_acquire_fence );
            }
        }
    }

    KA_TRACE( 20, ( "__kmp_hyper_barrier_release: T#%d(%d:%d) exit for barrier type %d\n",
      gtid, team->t.t_id, tid, bt ) );
}

/*
 * Internal function to do a barrier.
 * If is_split is true, do a split barrier, otherwise, do a plain barrier
 * If reduce is non-NULL, do a split reduction barrier, otherwise, do a split barrier
 * Returns 0 if master thread, 1 if worker thread.
 */
int
__kmp_barrier( enum barrier_type bt, int gtid, int is_split,
               size_t reduce_size, void *reduce_data, void (*reduce)(void *, void *) )
{
    register int          tid             = __kmp_tid_from_gtid( gtid );
    register kmp_info_t  *this_thr        = __kmp_threads[ gtid ];
    register kmp_team_t  *team            = this_thr -> th.th_team;
    register int status = 0;

    ident_t * tmp_loc = __kmp_threads[ gtid ]->th.th_ident;

    KA_TRACE( 15, ( "__kmp_barrier: T#%d(%d:%d) has arrived\n",
                    gtid, __kmp_team_from_gtid(gtid)->t.t_id, __kmp_tid_from_gtid(gtid) ) );

    if ( ! team->t.t_serialized ) {
#if USE_ITT_BUILD
        // This value will be used in itt notify events below.
        void * itt_sync_obj = NULL;
        #if USE_ITT_NOTIFY
            if ( __itt_sync_create_ptr || KMP_ITT_DEBUG )
                itt_sync_obj = __kmp_itt_barrier_object( gtid, bt, 1 );
        #endif
#endif /* USE_ITT_BUILD */
        #if OMP_30_ENABLED
            if ( __kmp_tasking_mode == tskm_extra_barrier ) {
                __kmp_tasking_barrier( team, this_thr, gtid );
                KA_TRACE( 15, ( "__kmp_barrier: T#%d(%d:%d) past tasking barrier\n",
                               gtid, __kmp_team_from_gtid(gtid)->t.t_id, __kmp_tid_from_gtid(gtid) ) );
            }
        #endif /* OMP_30_ENABLED */

        //
        // Copy the blocktime info to the thread, where __kmp_wait_sleep()
        // can access it when the team struct is not guaranteed to exist.
        //
        // See the note about the corresponding code in __kmp_join_barrier()
        // being performance-critical.
        //
        if ( __kmp_dflt_blocktime != KMP_MAX_BLOCKTIME ) {
            #if OMP_30_ENABLED
                this_thr -> th.th_team_bt_intervals = team -> t.t_implicit_task_taskdata[tid].td_icvs.bt_intervals;
                this_thr -> th.th_team_bt_set = team -> t.t_implicit_task_taskdata[tid].td_icvs.bt_set;
            #else
                this_thr -> th.th_team_bt_intervals = team -> t.t_set_bt_intervals[tid];
                this_thr -> th.th_team_bt_set= team -> t.t_set_bt_set[tid];
            #endif // OMP_30_ENABLED
        }

#if USE_ITT_BUILD
        if ( __itt_sync_create_ptr || KMP_ITT_DEBUG )
            __kmp_itt_barrier_starting( gtid, itt_sync_obj );
#endif /* USE_ITT_BUILD */

        if ( reduce != NULL ) {
            //KMP_DEBUG_ASSERT( is_split == TRUE );  // #C69956
            this_thr -> th.th_local.reduce_data = reduce_data;
        }
        if ( __kmp_barrier_gather_pattern[ bt ] == bp_linear_bar || __kmp_barrier_gather_branch_bits[ bt ] == 0 ) {
            __kmp_linear_barrier_gather( bt, this_thr, gtid, tid, reduce
                                         USE_ITT_BUILD_ARG( itt_sync_obj )
                                         );
        } else if ( __kmp_barrier_gather_pattern[ bt ] == bp_tree_bar ) {
            __kmp_tree_barrier_gather( bt, this_thr, gtid, tid, reduce
                                       USE_ITT_BUILD_ARG( itt_sync_obj )
                                       );
        } else {
            __kmp_hyper_barrier_gather( bt, this_thr, gtid, tid, reduce
                                        USE_ITT_BUILD_ARG( itt_sync_obj )
                                        );
        }; // if

#if USE_ITT_BUILD
        // TODO: In case of split reduction barrier, master thread may send acquired event early,
        // before the final summation into the shared variable is done (final summation can be a
        // long operation for array reductions).
        if ( __itt_sync_create_ptr || KMP_ITT_DEBUG )
            __kmp_itt_barrier_middle( gtid, itt_sync_obj );
#endif /* USE_ITT_BUILD */

        KMP_MB();

        if ( KMP_MASTER_TID( tid ) ) {
            status = 0;

            #if OMP_30_ENABLED
                if ( __kmp_tasking_mode != tskm_immediate_exec ) {
                    __kmp_task_team_wait(  this_thr, team
                                           USE_ITT_BUILD_ARG( itt_sync_obj )
                                           );
                    __kmp_task_team_setup( this_thr, team );
                }
            #endif /* OMP_30_ENABLED */


#if USE_ITT_BUILD && USE_ITT_NOTIFY
            // Barrier - report frame end
            if( __itt_frame_submit_v3_ptr && __kmp_forkjoin_frames_mode ) {
                kmp_uint64 tmp = __itt_get_timestamp();
                switch( __kmp_forkjoin_frames_mode ) {
                case 1:
                  __kmp_itt_frame_submit( gtid, this_thr->th.th_frame_time, tmp, 0, tmp_loc );
                  this_thr->th.th_frame_time = tmp;
                  break;
                case 2:
                  __kmp_itt_frame_submit( gtid, this_thr->th.th_bar_arrive_time, tmp, 1, tmp_loc );
                  break;
                case 3:
                  __kmp_itt_frame_submit( gtid, this_thr->th.th_frame_time, tmp, 0, tmp_loc );
                  __kmp_itt_frame_submit( gtid, this_thr->th.th_bar_arrive_time, tmp, 1, tmp_loc );
                  this_thr->th.th_frame_time = tmp;
                  break;
                }
            }
#endif /* USE_ITT_BUILD */
        } else {
            status = 1;
        }
        if ( status == 1 || ! is_split ) {
            if ( __kmp_barrier_release_pattern[ bt ] == bp_linear_bar || __kmp_barrier_release_branch_bits[ bt ] == 0 ) {
                __kmp_linear_barrier_release( bt, this_thr, gtid, tid, FALSE
                                              USE_ITT_BUILD_ARG( itt_sync_obj )
                                              );
            } else if ( __kmp_barrier_release_pattern[ bt ] == bp_tree_bar ) {
                __kmp_tree_barrier_release( bt, this_thr, gtid, tid, FALSE
                                            USE_ITT_BUILD_ARG( itt_sync_obj )
                                            );
            } else {
                __kmp_hyper_barrier_release( bt, this_thr, gtid, tid, FALSE
                                             USE_ITT_BUILD_ARG( itt_sync_obj )
                                             );
            }
            #if OMP_30_ENABLED
                if ( __kmp_tasking_mode != tskm_immediate_exec ) {
                    __kmp_task_team_sync( this_thr, team );
                }
            #endif /* OMP_30_ENABLED */
        }

#if USE_ITT_BUILD
        // GEH: TODO: Move this under if-condition above and also include in __kmp_end_split_barrier().
        //      This will more accurately represent the actual release time of the threads for split barriers.
        if ( __itt_sync_create_ptr || KMP_ITT_DEBUG )
            __kmp_itt_barrier_finished( gtid, itt_sync_obj );
#endif /* USE_ITT_BUILD */

    } else {    // Team is serialized.

        status = 0;

        #if OMP_30_ENABLED
            if ( __kmp_tasking_mode != tskm_immediate_exec ) {
                //
                // The task team should be NULL for serialized code.
                // (tasks will be executed immediately).
                //
                KMP_DEBUG_ASSERT( team->t.t_task_team == NULL );
                KMP_DEBUG_ASSERT( this_thr->th.th_task_team == NULL );
            }
        #endif /* OMP_30_ENABLED */
    }

    KA_TRACE( 15, ( "__kmp_barrier: T#%d(%d:%d) is leaving with return value %d\n",
                    gtid, __kmp_team_from_gtid(gtid)->t.t_id, __kmp_tid_from_gtid(gtid),
                    status ) );
    return status;
}


void
__kmp_end_split_barrier( enum barrier_type bt, int gtid )
{
    int         tid      = __kmp_tid_from_gtid( gtid );
    kmp_info_t *this_thr = __kmp_threads[ gtid ];
    kmp_team_t *team     = this_thr -> th.th_team;

    if( ! team -> t.t_serialized ) {
        if( KMP_MASTER_GTID( gtid ) ) {
            if ( __kmp_barrier_release_pattern[ bt ] == bp_linear_bar || __kmp_barrier_release_branch_bits[ bt ] == 0 ) {
                __kmp_linear_barrier_release( bt, this_thr, gtid, tid, FALSE
#if USE_ITT_BUILD
                                              , NULL
#endif /* USE_ITT_BUILD */
                                              );
            } else if ( __kmp_barrier_release_pattern[ bt ] == bp_tree_bar ) {
                __kmp_tree_barrier_release( bt, this_thr, gtid, tid, FALSE
#if USE_ITT_BUILD
                                            , NULL
#endif /* USE_ITT_BUILD */
                                            );
            } else {
                __kmp_hyper_barrier_release( bt, this_thr, gtid, tid, FALSE
#if USE_ITT_BUILD
                                             , NULL
#endif /* USE_ITT_BUILD */
                                             );
            }; // if
            #if OMP_30_ENABLED
                if ( __kmp_tasking_mode != tskm_immediate_exec ) {
                    __kmp_task_team_sync( this_thr, team );
                }; // if
            #endif /* OMP_30_ENABLED */
        }
    }
}

/* ------------------------------------------------------------------------ */
/* ------------------------------------------------------------------------ */

/*
 * determine if we can go parallel or must use a serialized parallel region and
 * how many threads we can use
 * set_nproc is the number of threads requested for the team
 * returns 0 if we should serialize or only use one thread,
 * otherwise the number of threads to use
 * The forkjoin lock is held by the caller.
 */
static int
__kmp_reserve_threads( kmp_root_t *root, kmp_team_t *parent_team,
   int master_tid, int set_nthreads
#if OMP_40_ENABLED
  , int enter_teams
#endif /* OMP_40_ENABLED */
)
{
    int capacity;
    int new_nthreads;
    int use_rml_to_adjust_nth;
    KMP_DEBUG_ASSERT( __kmp_init_serial );
    KMP_DEBUG_ASSERT( root && parent_team );

    //
    // Initial check to see if we should use a serialized team.
    //
    if ( set_nthreads == 1 ) {
        KC_TRACE( 10, ( "__kmp_reserve_threads: T#%d reserving 1 thread; requested %d threads\n",
                        __kmp_get_gtid(), set_nthreads ));
        return 1;
    }
    if ( ( !get__nested_2(parent_team,master_tid) && (root->r.r_in_parallel
#if OMP_40_ENABLED
       && !enter_teams
#endif /* OMP_40_ENABLED */
       ) ) || ( __kmp_library == library_serial ) ) {
        KC_TRACE( 10, ( "__kmp_reserve_threads: T#%d serializing team; requested %d threads\n",
                        __kmp_get_gtid(), set_nthreads ));
        return 1;
    }

    //
    // If dyn-var is set, dynamically adjust the number of desired threads,
    // according to the method specified by dynamic_mode.
    //
    new_nthreads = set_nthreads;
    use_rml_to_adjust_nth = FALSE;
    if ( ! get__dynamic_2( parent_team, master_tid ) ) {
        ;
    }
#ifdef USE_LOAD_BALANCE
    else if ( __kmp_global.g.g_dynamic_mode == dynamic_load_balance ) {
        new_nthreads = __kmp_load_balance_nproc( root, set_nthreads );
        if ( new_nthreads == 1 ) {
            KC_TRACE( 10, ( "__kmp_reserve_threads: T#%d load balance reduced reservation to 1 thread\n",
              master_tid ));
            return 1;
        }
        if ( new_nthreads < set_nthreads ) {
            KC_TRACE( 10, ( "__kmp_reserve_threads: T#%d load balance reduced reservation to %d threads\n",
              master_tid, new_nthreads ));
        }
    }
#endif /* USE_LOAD_BALANCE */
    else if ( __kmp_global.g.g_dynamic_mode == dynamic_thread_limit ) {
        new_nthreads = __kmp_avail_proc - __kmp_nth + (root->r.r_active ? 1
          : root->r.r_hot_team->t.t_nproc);
        if ( new_nthreads <= 1 ) {
            KC_TRACE( 10, ( "__kmp_reserve_threads: T#%d thread limit reduced reservation to 1 thread\n",
              master_tid ));
            return 1;
        }
        if ( new_nthreads < set_nthreads ) {
            KC_TRACE( 10, ( "__kmp_reserve_threads: T#%d thread limit reduced reservation to %d threads\n",
              master_tid, new_nthreads ));
        }
        else {
            new_nthreads = set_nthreads;
        }
    }
    else if ( __kmp_global.g.g_dynamic_mode == dynamic_random ) {
        if ( set_nthreads > 2 ) {
            new_nthreads = __kmp_get_random( parent_team->t.t_threads[master_tid] );
            new_nthreads = ( new_nthreads % set_nthreads ) + 1;
            if ( new_nthreads == 1 ) {
                KC_TRACE( 10, ( "__kmp_reserve_threads: T#%d dynamic random reduced reservation to 1 thread\n",
                  master_tid ));
                return 1;
            }
            if ( new_nthreads < set_nthreads ) {
                KC_TRACE( 10, ( "__kmp_reserve_threads: T#%d dynamic random reduced reservation to %d threads\n",
                  master_tid, new_nthreads ));
            }
        }
    }
    else {
        KMP_ASSERT( 0 );
    }

    //
    // Respect KMP_ALL_THREADS, KMP_MAX_THREADS, OMP_THREAD_LIMIT.
    //
    if ( __kmp_nth + new_nthreads - ( root->r.r_active ? 1 :
      root->r.r_hot_team->t.t_nproc ) > __kmp_max_nth ) {
        int tl_nthreads = __kmp_max_nth - __kmp_nth + ( root->r.r_active ? 1 :
          root->r.r_hot_team->t.t_nproc );
        if ( tl_nthreads <= 0 ) {
            tl_nthreads = 1;
        }

        //
        // If dyn-var is false, emit a 1-time warning.
        //
        if ( ! get__dynamic_2( parent_team, master_tid )
          && ( ! __kmp_reserve_warn ) ) {
            __kmp_reserve_warn = 1;
            __kmp_msg(
                kmp_ms_warning,
                KMP_MSG( CantFormThrTeam, set_nthreads, tl_nthreads ),
                KMP_HNT( Unset_ALL_THREADS ),
                __kmp_msg_null
            );
        }
        if ( tl_nthreads == 1 ) {
            KC_TRACE( 10, ( "__kmp_reserve_threads: T#%d KMP_ALL_THREADS reduced reservation to 1 thread\n",
              master_tid ));
            return 1;
        }
        KC_TRACE( 10, ( "__kmp_reserve_threads: T#%d KMP_ALL_THREADS reduced reservation to %d threads\n",
          master_tid, tl_nthreads ));
        new_nthreads = tl_nthreads;
    }


    //
    // Check if the threads array is large enough, or needs expanding.
    //
    // See comment in __kmp_register_root() about the adjustment if
    // __kmp_threads[0] == NULL.
    //
    capacity = __kmp_threads_capacity;
    if ( TCR_PTR(__kmp_threads[0]) == NULL ) {
        --capacity;
    }
    if ( __kmp_nth + new_nthreads - ( root->r.r_active ? 1 :
      root->r.r_hot_team->t.t_nproc ) > capacity ) {
        //
        // Expand the threads array.
        //
        int slotsRequired = __kmp_nth + new_nthreads - ( root->r.r_active ? 1 :
          root->r.r_hot_team->t.t_nproc ) - capacity;
        int slotsAdded = __kmp_expand_threads(slotsRequired, slotsRequired);
        if ( slotsAdded < slotsRequired ) {
            //
            // The threads array was not expanded enough.
            //
            new_nthreads -= ( slotsRequired - slotsAdded );
            KMP_ASSERT( new_nthreads >= 1 );

            //
            // If dyn-var is false, emit a 1-time warning.
            //
            if ( ! get__dynamic_2( parent_team, master_tid )
              && ( ! __kmp_reserve_warn ) ) {
                __kmp_reserve_warn = 1;
                if ( __kmp_tp_cached ) {
                    __kmp_msg(
                        kmp_ms_warning,
                        KMP_MSG( CantFormThrTeam, set_nthreads, new_nthreads ),
                        KMP_HNT( Set_ALL_THREADPRIVATE, __kmp_tp_capacity ),
                        KMP_HNT( PossibleSystemLimitOnThreads ),
                        __kmp_msg_null
                    );
                }
                else {
                    __kmp_msg(
                        kmp_ms_warning,
                        KMP_MSG( CantFormThrTeam, set_nthreads, new_nthreads ),
                        KMP_HNT( SystemLimitOnThreads ),
                        __kmp_msg_null
                    );
                }
            }
        }
    }

    if ( new_nthreads == 1 ) {
        KC_TRACE( 10, ( "__kmp_reserve_threads: T#%d serializing team after reclaiming dead roots and rechecking; requested %d threads\n",
                        __kmp_get_gtid(), set_nthreads ) );
        return 1;
    }

    KC_TRACE( 10, ( "__kmp_reserve_threads: T#%d allocating %d threads; requested %d threads\n",
                    __kmp_get_gtid(), new_nthreads, set_nthreads ));
    return new_nthreads;
}

/* ------------------------------------------------------------------------ */
/* ------------------------------------------------------------------------ */

/* allocate threads from the thread pool and assign them to the new team */
/* we are assured that there are enough threads available, because we
 * checked on that earlier within critical section forkjoin */

static void
__kmp_fork_team_threads( kmp_root_t *root, kmp_team_t *team,
                         kmp_info_t *master_th, int master_gtid )
{
    int         i;

    KA_TRACE( 10, ("__kmp_fork_team_threads: new_nprocs = %d\n", team->t.t_nproc ) );
    KMP_DEBUG_ASSERT( master_gtid == __kmp_get_gtid() );
    KMP_MB();

    /* first, let's setup the master thread */
    master_th -> th.th_info.ds.ds_tid  = 0;
    master_th -> th.th_team            = team;
    master_th -> th.th_team_nproc      = team -> t.t_nproc;
    master_th -> th.th_team_master     = master_th;
    master_th -> th.th_team_serialized = FALSE;
    master_th -> th.th_dispatch        = & team -> t.t_dispatch[ 0 ];

    /* make sure we are not the optimized hot team */
    if ( team != root->r.r_hot_team ) {

        /* install the master thread */
        team -> t.t_threads[ 0 ]    = master_th;
        __kmp_initialize_info( master_th, team, 0, master_gtid );

        /* now, install the worker threads */
        for ( i=1 ;  i < team->t.t_nproc ; i++ ) {

            /* fork or reallocate a new thread and install it in team */
            team -> t.t_threads[ i ] =  __kmp_allocate_thread( root, team, i );
            KMP_DEBUG_ASSERT( team->t.t_threads[i] );
            KMP_DEBUG_ASSERT( team->t.t_threads[i]->th.th_team == team );
            /* align team and thread arrived states */
            KA_TRACE( 20, ("__kmp_fork_team_threads: T#%d(%d:%d) init arrived T#%d(%d:%d) join =%u, plain=%u\n",
                            __kmp_gtid_from_tid( 0, team ), team->t.t_id, 0,
                            __kmp_gtid_from_tid( i, team ), team->t.t_id, i,
                            team->t.t_bar[ bs_forkjoin_barrier ].b_arrived,
                            team->t.t_bar[ bs_plain_barrier ].b_arrived ) );

            { // Initialize threads' barrier data.
                int b;
                kmp_balign_t * balign = team->t.t_threads[ i ]->th.th_bar;
                for ( b = 0; b < bs_last_barrier; ++ b ) {
                    balign[ b ].bb.b_arrived        = team->t.t_bar[ b ].b_arrived;
                }; // for b
            }
        }

#if OMP_40_ENABLED && KMP_AFFINITY_SUPPORTED
        __kmp_partition_places( team );
#endif

    }

    KMP_MB();
}

static void
__kmp_alloc_argv_entries( int argc, kmp_team_t *team, int realloc ); // forward declaration

static void
__kmp_setup_icv_copy( kmp_team_t *team, int new_nproc,
#if OMP_30_ENABLED
                 kmp_internal_control_t * new_icvs,
                 ident_t *                loc
#else
                 int new_set_nproc, int new_set_dynamic, int new_set_nested,
                 int new_set_blocktime, int new_bt_intervals, int new_bt_set
#endif // OMP_30_ENABLED
                 ); // forward declaration

/* most of the work for a fork */
/* return true if we really went parallel, false if serialized */
int
__kmp_fork_call(
    ident_t   * loc,
    int         gtid,
    int         exec_master, // 0 - GNU native code, master doesn't invoke microtask
                             // 1 - Intel code, master invokes microtask
                             // 2 - MS native code, use special invoker
    kmp_int32   argc,
    microtask_t microtask,
    launch_t    invoker,
/* TODO: revert workaround for Intel(R) 64 tracker #96 */
#if (KMP_ARCH_X86_64 || KMP_ARCH_ARM) && KMP_OS_LINUX
    va_list   * ap
#else
    va_list     ap
#endif
    )
{
    void          **argv;
    int             i;
    int             master_tid;
    int             master_this_cons;
    int             master_last_cons;
    kmp_team_t     *team;
    kmp_team_t     *parent_team;
    kmp_info_t     *master_th;
    kmp_root_t     *root;
    int             nthreads;
    int             master_active;
    int             master_set_numthreads;
    int             level;
#if OMP_40_ENABLED
    int             teams_level;
#endif

    KA_TRACE( 20, ("__kmp_fork_call: enter T#%d\n", gtid ));

    /* initialize if needed */
    KMP_DEBUG_ASSERT( __kmp_init_serial );
    if( ! TCR_4(__kmp_init_parallel) )
        __kmp_parallel_initialize();

    /* setup current data */
    master_th     = __kmp_threads[ gtid ];
    parent_team   = master_th -> th.th_team;
    master_tid    = master_th -> th.th_info.ds.ds_tid;
    master_this_cons = master_th -> th.th_local.this_construct;
    master_last_cons = master_th -> th.th_local.last_construct;
    root          = master_th -> th.th_root;
    master_active = root -> r.r_active;
    master_set_numthreads = master_th -> th.th_set_nproc;
#if OMP_30_ENABLED
    // Nested level will be an index in the nested nthreads array
    level         = parent_team->t.t_level;
#endif // OMP_30_ENABLED
#if OMP_40_ENABLED
    teams_level    = master_th->th.th_teams_level; // needed to check nesting inside the teams
#endif


    master_th->th.th_ident = loc;

#if OMP_40_ENABLED
    if ( master_th->th.th_team_microtask &&
         ap && microtask != (microtask_t)__kmp_teams_master && level == teams_level ) {
        // AC: This is start of parallel that is nested inside teams construct.
        //     The team is actual (hot), all workers are ready at the fork barrier.
        //     No lock needed to initialize the team a bit, then free workers.
        parent_team->t.t_ident = loc;
        parent_team->t.t_argc  = argc;
        argv = (void**)parent_team->t.t_argv;
        for( i=argc-1; i >= 0; --i )
/* TODO: revert workaround for Intel(R) 64 tracker #96 */
#if (KMP_ARCH_X86_64 || KMP_ARCH_ARM) && KMP_OS_LINUX
            *argv++ = va_arg( *ap, void * );
#else
            *argv++ = va_arg( ap, void * );
#endif
        /* Increment our nested depth levels, but not increase the serialization */
        if ( parent_team == master_th->th.th_serial_team ) {
            // AC: we are in serialized parallel
            __kmpc_serialized_parallel(loc, gtid);
            KMP_DEBUG_ASSERT( parent_team->t.t_serialized > 1 );
            parent_team->t.t_serialized--; // AC: need this in order enquiry functions
                                           //     work correctly, will restore at join time
            __kmp_invoke_microtask( microtask, gtid, 0, argc, parent_team->t.t_argv );
            return TRUE;
        }
        parent_team->t.t_pkfn  = microtask;
        parent_team->t.t_invoke = invoker;
        KMP_TEST_THEN_INC32( (kmp_int32*) &root->r.r_in_parallel );
        parent_team->t.t_active_level ++;
        parent_team->t.t_level ++;

        /* Change number of threads in the team if requested */
        if ( master_set_numthreads ) {   // The parallel has num_threads clause
            if ( master_set_numthreads < master_th->th.th_set_nth_teams ) {
                // AC: only can reduce the number of threads dynamically, cannot increase
                kmp_info_t **other_threads = parent_team->t.t_threads;
                parent_team->t.t_nproc = master_set_numthreads;
                for ( i = 0; i < master_set_numthreads; ++i ) {
                    other_threads[i]->th.th_team_nproc = master_set_numthreads;
                }
                // Keep extra threads hot in the team for possible next parallels
            }
            master_th->th.th_set_nproc = 0;
        }


        KF_TRACE( 10, ( "__kmp_fork_call: before internal fork: root=%p, team=%p, master_th=%p, gtid=%d\n", root, parent_team, master_th, gtid ) );
        __kmp_internal_fork( loc, gtid, parent_team );
        KF_TRACE( 10, ( "__kmp_fork_call: after internal fork: root=%p, team=%p, master_th=%p, gtid=%d\n", root, parent_team, master_th, gtid ) );

        /* Invoke microtask for MASTER thread */
        KA_TRACE( 20, ("__kmp_fork_call: T#%d(%d:0) invoke microtask = %p\n",
                    gtid, parent_team->t.t_id, parent_team->t.t_pkfn ) );

        if (! parent_team->t.t_invoke( gtid )) {
            KMP_ASSERT2( 0, "cannot invoke microtask for MASTER thread" );
        }
        KA_TRACE( 20, ("__kmp_fork_call: T#%d(%d:0) done microtask = %p\n",
            gtid, parent_team->t.t_id, parent_team->t.t_pkfn ) );
        KMP_MB();       /* Flush all pending memory write invalidates.  */

        KA_TRACE( 20, ("__kmp_fork_call: parallel exit T#%d\n", gtid ));

        return TRUE;
    }
#endif /* OMP_40_ENABLED */

#if OMP_30_ENABLED && KMP_DEBUG
    if ( __kmp_tasking_mode != tskm_immediate_exec ) {
        KMP_DEBUG_ASSERT( master_th->th.th_task_team == parent_team->t.t_task_team );
    }
#endif // OMP_30_ENABLED

    /* determine how many new threads we can use */
    __kmp_acquire_bootstrap_lock( &__kmp_forkjoin_lock );

#if OMP_30_ENABLED
    if ( parent_team->t.t_active_level >= master_th->th.th_current_task->td_icvs.max_active_levels ) {
        nthreads = 1;
    }
    else
#endif // OMP_30_ENABLED

    {
        nthreads = master_set_numthreads ?
            master_set_numthreads : get__nproc_2( parent_team, master_tid );
        nthreads = __kmp_reserve_threads( root, parent_team, master_tid, nthreads
#if OMP_40_ENABLED
        // AC: If we execute teams from parallel region (on host), then teams
        //     should be created but each can only have 1 thread if nesting is disabled.
        //     If teams called from serial region, then teams and their threads
        //     should be created regardless of the nesting setting.
                                        ,( ( ap == NULL && teams_level == 0 ) ||
                                           ( ap && teams_level > 0 && teams_level == level ) )
#endif /* OMP_40_ENABLED */
        );
    }
    KMP_DEBUG_ASSERT( nthreads > 0 );

    /* If we temporarily changed the set number of threads then restore it now */
    master_th -> th.th_set_nproc = 0;


    /* create a serialized parallel region? */
    if ( nthreads == 1 ) {
        /* josh todo: hypothetical question: what do we do for OS X*? */
#if KMP_OS_LINUX && ( KMP_ARCH_X86 || KMP_ARCH_X86_64 || KMP_ARCH_ARM )
        void *   args[ argc ];
#else
        void * * args = (void**) alloca( argc * sizeof( void * ) );
#endif /* KMP_OS_LINUX && ( KMP_ARCH_X86 || KMP_ARCH_X86_64 || KMP_ARCH_ARM ) */

        __kmp_release_bootstrap_lock( &__kmp_forkjoin_lock );
        KA_TRACE( 20, ("__kmp_fork_call: T#%d serializing parallel region\n", gtid ));

        __kmpc_serialized_parallel(loc, gtid);

        if ( exec_master == 0 ) {
            // we were called from GNU native code
            KA_TRACE( 20, ("__kmp_fork_call: T#%d serial exit\n", gtid ));
            return FALSE;
        } else if ( exec_master == 1 ) {
            /* TODO this sucks, use the compiler itself to pass args! :) */
            master_th -> th.th_serial_team -> t.t_ident =  loc;
#if OMP_40_ENABLED
            if ( !ap ) {
                // revert change made in __kmpc_serialized_parallel()
                master_th -> th.th_serial_team -> t.t_level--;
                // Get args from parent team for teams construct
                __kmp_invoke_microtask( microtask, gtid, 0, argc, parent_team->t.t_argv );
            } else if ( microtask == (microtask_t)__kmp_teams_master ) {
                KMP_DEBUG_ASSERT( master_th->th.th_team == master_th->th.th_serial_team );
                team = master_th->th.th_team;
                //team->t.t_pkfn = microtask;
                team->t.t_invoke = invoker;
                __kmp_alloc_argv_entries( argc, team, TRUE );
                team->t.t_argc = argc;
                argv = (void**) team->t.t_argv;
                if ( ap ) {
                    for( i=argc-1; i >= 0; --i )
                      /* TODO: revert workaround for Intel(R) 64 tracker #96 */
                      #if (KMP_ARCH_X86_64 || KMP_ARCH_ARM) && KMP_OS_LINUX
                        *argv++ = va_arg( *ap, void * );
                      #else
                        *argv++ = va_arg( ap, void * );
                      #endif
                } else {
                    for( i=0; i < argc; ++i )
                        // Get args from parent team for teams construct
                        argv[i] = parent_team->t.t_argv[i];
                }
                // AC: revert change made in __kmpc_serialized_parallel()
                //     because initial code in teams should have level=0
                team->t.t_level--;
                // AC: call special invoker for outer "parallel" of the teams construct
                invoker(gtid);
            } else {
#endif /* OMP_40_ENABLED */
                argv = args;
                for( i=argc-1; i >= 0; --i )
                /* TODO: revert workaround for Intel(R) 64 tracker #96 */
                #if (KMP_ARCH_X86_64 || KMP_ARCH_ARM) && KMP_OS_LINUX
                    *argv++ = va_arg( *ap, void * );
                #else
                    *argv++ = va_arg( ap, void * );
                #endif
                KMP_MB();
                __kmp_invoke_microtask( microtask, gtid, 0, argc, args );
#if OMP_40_ENABLED
            }
#endif /* OMP_40_ENABLED */
        }
        else {
            KMP_ASSERT2( exec_master <= 1, "__kmp_fork_call: unknown parameter exec_master" );
        }

        KA_TRACE( 20, ("__kmp_fork_call: T#%d serial exit\n", gtid ));

        KMP_MB();
        return FALSE;
    }

#if OMP_30_ENABLED
    // GEH: only modify the executing flag in the case when not serialized
    //      serialized case is handled in kmpc_serialized_parallel
    KF_TRACE( 10, ( "__kmp_fork_call: parent_team_aclevel=%d, master_th=%p, curtask=%p, curtask_max_aclevel=%d\n",
                    parent_team->t.t_active_level, master_th, master_th->th.th_current_task,
                    master_th->th.th_current_task->td_icvs.max_active_levels ) );
    // TODO: GEH - cannot do this assertion because root thread not set up as executing
    // KMP_ASSERT( master_th->th.th_current_task->td_flags.executing == 1 );
    master_th->th.th_current_task->td_flags.executing = 0;
#endif

#if OMP_40_ENABLED
    if ( !master_th->th.th_team_microtask || level > teams_level )
#endif /* OMP_40_ENABLED */
    {
        /* Increment our nested depth level */
        KMP_TEST_THEN_INC32( (kmp_int32*) &root->r.r_in_parallel );
    }

#if OMP_30_ENABLED
    //
    // See if we need to make a copy of the ICVs.
    //
    int nthreads_icv = master_th->th.th_current_task->td_icvs.nproc;
    if ( ( level + 1 < __kmp_nested_nth.used ) &&
      ( __kmp_nested_nth.nth[level + 1] != nthreads_icv ) ) {
        nthreads_icv = __kmp_nested_nth.nth[level + 1];
    }
    else {
        nthreads_icv = 0;  // don't update
    }

#if OMP_40_ENABLED
    //
    // Figure out the proc_bind_policy for the new team.
    //
    kmp_proc_bind_t proc_bind = master_th->th.th_set_proc_bind;
    kmp_proc_bind_t proc_bind_icv; // proc_bind_default means don't update

    if ( master_th->th.th_current_task->td_icvs.proc_bind == proc_bind_false ) {
        proc_bind = proc_bind_false;
        proc_bind_icv = proc_bind_default;
    }
    else {
        proc_bind_icv = master_th->th.th_current_task->td_icvs.proc_bind;
        if ( proc_bind == proc_bind_default ) {
            //
            // No proc_bind clause was specified, so use the current value
            // of proc-bind-var for this parallel region.
            //
            proc_bind = proc_bind_icv;
        }
        else {
            //
            // The proc_bind policy was specified explicitly on the parallel
            // clause.  This overrides the proc-bind-var for this parallel
            // region, but does not change proc-bind-var.
            //
        }

        //
        // Figure the value of proc-bind-var for the child threads.
        //
        if ( ( level + 1 < __kmp_nested_proc_bind.used )
          && ( __kmp_nested_proc_bind.bind_types[level + 1] != proc_bind_icv ) ) {
            proc_bind_icv = __kmp_nested_proc_bind.bind_types[level + 1];
        }
        else {
            proc_bind_icv = proc_bind_default;
        }
    }

    //
    // Reset for next parallel region
    //
    master_th->th.th_set_proc_bind = proc_bind_default;
#endif /* OMP_40_ENABLED */

    if ( ( nthreads_icv > 0 )
#if OMP_40_ENABLED
      || ( proc_bind_icv != proc_bind_default )
#endif /* OMP_40_ENABLED */
      )
    {
        kmp_internal_control_t new_icvs;
        copy_icvs( & new_icvs, & master_th->th.th_current_task->td_icvs );
        new_icvs.next = NULL;

        if ( nthreads_icv > 0 ) {
            new_icvs.nproc = nthreads_icv;
        }

#if OMP_40_ENABLED
        if ( proc_bind_icv != proc_bind_default ) {
            new_icvs.proc_bind = proc_bind_icv;
        }
#endif /* OMP_40_ENABLED */

        /* allocate a new parallel team */
        KF_TRACE( 10, ( "__kmp_fork_call: before __kmp_allocate_team\n" ) );
        team = __kmp_allocate_team(root, nthreads, nthreads,
#if OMP_40_ENABLED
          proc_bind,
#endif
          &new_icvs, argc );
    } else
#endif /* OMP_30_ENABLED */
    {
        /* allocate a new parallel team */
        KF_TRACE( 10, ( "__kmp_fork_call: before __kmp_allocate_team\n" ) );
        team = __kmp_allocate_team(root, nthreads, nthreads,
#if OMP_40_ENABLED
                proc_bind,
#endif
#if OMP_30_ENABLED
                &master_th->th.th_current_task->td_icvs,
#else
                parent_team->t.t_set_nproc[master_tid],
                parent_team->t.t_set_dynamic[master_tid],
                parent_team->t.t_set_nested[master_tid],
                parent_team->t.t_set_blocktime[master_tid],
                parent_team->t.t_set_bt_intervals[master_tid],
                parent_team->t.t_set_bt_set[master_tid],
#endif // OMP_30_ENABLED
                argc );
    }

    KF_TRACE( 10, ( "__kmp_fork_call: after __kmp_allocate_team - team = %p\n",
            team ) );

    /* setup the new team */
    team->t.t_master_tid = master_tid;
    team->t.t_master_this_cons = master_this_cons;
    team->t.t_master_last_cons = master_last_cons;

    team->t.t_parent     = parent_team;
    TCW_SYNC_PTR(team->t.t_pkfn, microtask);
    team->t.t_invoke     = invoker;  /* TODO move this to root, maybe */
    team->t.t_ident      = loc;
#if OMP_30_ENABLED
    // TODO: parent_team->t.t_level == INT_MAX ???
#if OMP_40_ENABLED
    if ( !master_th->th.th_team_microtask || level > teams_level ) {
#endif /* OMP_40_ENABLED */
        team->t.t_level        = parent_team->t.t_level + 1;
        team->t.t_active_level = parent_team->t.t_active_level + 1;
#if OMP_40_ENABLED
    } else {
        // AC: Do not increase parallel level at start of the teams construct
        team->t.t_level        = parent_team->t.t_level;
        team->t.t_active_level = parent_team->t.t_active_level;
    }
#endif /* OMP_40_ENABLED */
    team->t.t_sched      = get__sched_2( parent_team, master_tid ); // set master's schedule as new run-time schedule

#if KMP_ARCH_X86 || KMP_ARCH_X86_64
    if ( __kmp_inherit_fp_control ) {
        __kmp_store_x87_fpu_control_word( &team->t.t_x87_fpu_control_word );
        __kmp_store_mxcsr( &team->t.t_mxcsr );
        team->t.t_mxcsr &= KMP_X86_MXCSR_MASK;
        team->t.t_fp_control_saved = TRUE;
    }
    else {
        team->t.t_fp_control_saved = FALSE;
    }
#endif /* KMP_ARCH_X86 || KMP_ARCH_X86_64 */

    if ( __kmp_tasking_mode != tskm_immediate_exec ) {
        //
        // Set the master thread's task team to the team's task team.
        // Unless this is the hot team, it should be NULL.
        //
        KMP_DEBUG_ASSERT( master_th->th.th_task_team == parent_team->t.t_task_team );
        KA_TRACE( 20, ( "__kmp_fork_call: Master T#%d pushing task_team %p / team %p, new task_team %p / team %p\n",
                        __kmp_gtid_from_thread( master_th ), master_th->th.th_task_team,
                        parent_team, team->t.t_task_team, team ) );
        master_th->th.th_task_team = team->t.t_task_team;
        KMP_DEBUG_ASSERT( ( master_th->th.th_task_team == NULL ) || ( team == root->r.r_hot_team ) ) ;
    }
#endif // OMP_30_ENABLED

    KA_TRACE( 20, ("__kmp_fork_call: T#%d(%d:%d)->(%d:0) created a team of %d threads\n",
                gtid, parent_team->t.t_id, team->t.t_master_tid, team->t.t_id, team->t.t_nproc ));
    KMP_DEBUG_ASSERT( team != root->r.r_hot_team ||
                      ( team->t.t_master_tid == 0 &&
                        ( team->t.t_parent == root->r.r_root_team || team->t.t_parent->t.t_serialized ) ));
    KMP_MB();

    /* now, setup the arguments */
    argv = (void**) team -> t.t_argv;
#if OMP_40_ENABLED
    if ( ap ) {
#endif /* OMP_40_ENABLED */
        for( i=argc-1; i >= 0; --i )
/* TODO: revert workaround for Intel(R) 64 tracker #96 */
#if (KMP_ARCH_X86_64 || KMP_ARCH_ARM) && KMP_OS_LINUX
            *argv++ = va_arg( *ap, void * );
#else
            *argv++ = va_arg( ap, void * );
#endif
#if OMP_40_ENABLED
    } else {
        for( i=0; i < argc; ++i )
            // Get args from parent team for teams construct
            argv[i] = team->t.t_parent->t.t_argv[i];
    }
#endif /* OMP_40_ENABLED */

    /* now actually fork the threads */

    team->t.t_master_active = master_active;
    if (!root -> r.r_active)  /* Only do the assignment if it makes a difference to prevent cache ping-pong */
        root -> r.r_active = TRUE;

    __kmp_fork_team_threads( root, team, master_th, gtid );
    __kmp_setup_icv_copy(team, nthreads
#if OMP_30_ENABLED
			 , &master_th->th.th_current_task->td_icvs, loc
#else
			 , parent_team->t.t_set_nproc[master_tid],
			 parent_team->t.t_set_dynamic[master_tid],
			 parent_team->t.t_set_nested[master_tid],
			 parent_team->t.t_set_blocktime[master_tid],
			 parent_team->t.t_set_bt_intervals[master_tid],
			 parent_team->t.t_set_bt_set[master_tid]
#endif /* OMP_30_ENABLED */
			 );


    __kmp_release_bootstrap_lock( &__kmp_forkjoin_lock );


#if USE_ITT_BUILD
    // Mark start of "parallel" region for VTune. Only use one of frame notification scheme at the moment.
    if ( ( __itt_frame_begin_v3_ptr && __kmp_forkjoin_frames && ! __kmp_forkjoin_frames_mode ) || KMP_ITT_DEBUG )
# if OMP_40_ENABLED
    if ( !master_th->th.th_team_microtask || microtask == (microtask_t)__kmp_teams_master )
        // Either not in teams or the outer fork of the teams construct
# endif /* OMP_40_ENABLED */
        __kmp_itt_region_forking( gtid );
#endif /* USE_ITT_BUILD */

#if USE_ITT_BUILD && USE_ITT_NOTIFY && OMP_30_ENABLED
    // Internal fork - report frame begin
    if( ( __kmp_forkjoin_frames_mode == 1 || __kmp_forkjoin_frames_mode == 3 ) && __itt_frame_submit_v3_ptr && __itt_get_timestamp_ptr )
    {
        if( ! ( team->t.t_active_level > 1 ) ) {
            master_th->th.th_frame_time   = __itt_get_timestamp();
        }
    }
#endif /* USE_ITT_BUILD */

    /* now go on and do the work */
    KMP_DEBUG_ASSERT( team == __kmp_threads[gtid]->th.th_team );
    KMP_MB();

    KF_TRACE( 10, ( "__kmp_internal_fork : root=%p, team=%p, master_th=%p, gtid=%d\n", root, team, master_th, gtid ) );

#if USE_ITT_BUILD
    if ( __itt_stack_caller_create_ptr ) {
        team->t.t_stack_id = __kmp_itt_stack_caller_create(); // create new stack stitching id before entering fork barrier
    }
#endif /* USE_ITT_BUILD */

#if OMP_40_ENABLED
    if ( ap )   // AC: skip __kmp_internal_fork at teams construct, let only master threads execute
#endif /* OMP_40_ENABLED */
    {
        __kmp_internal_fork( loc, gtid, team );
        KF_TRACE( 10, ( "__kmp_internal_fork : after : root=%p, team=%p, master_th=%p, gtid=%d\n", root, team, master_th, gtid ) );
    }

    if (! exec_master) {
        KA_TRACE( 20, ("__kmp_fork_call: parallel exit T#%d\n", gtid ));
        return TRUE;
    }

    /* Invoke microtask for MASTER thread */
    KA_TRACE( 20, ("__kmp_fork_call: T#%d(%d:0) invoke microtask = %p\n",
                gtid, team->t.t_id, team->t.t_pkfn ) );

    if (! team->t.t_invoke( gtid )) {
        KMP_ASSERT2( 0, "cannot invoke microtask for MASTER thread" );
    }
    KA_TRACE( 20, ("__kmp_fork_call: T#%d(%d:0) done microtask = %p\n",
        gtid, team->t.t_id, team->t.t_pkfn ) );
    KMP_MB();       /* Flush all pending memory write invalidates.  */

    KA_TRACE( 20, ("__kmp_fork_call: parallel exit T#%d\n", gtid ));

    return TRUE;
}


void
__kmp_join_call(ident_t *loc, int gtid
#if OMP_40_ENABLED
               , int exit_teams
#endif /* OMP_40_ENABLED */
)
{
    kmp_team_t     *team;
    kmp_team_t     *parent_team;
    kmp_info_t     *master_th;
    kmp_root_t     *root;
    int             master_active;
    int             i;

    KA_TRACE( 20, ("__kmp_join_call: enter T#%d\n", gtid ));

    /* setup current data */
    master_th     = __kmp_threads[ gtid ];
    root          = master_th -> th.th_root;
    team          = master_th -> th.th_team;
    parent_team   = team->t.t_parent;

    master_th->th.th_ident = loc;

#if OMP_30_ENABLED && KMP_DEBUG
    if ( __kmp_tasking_mode != tskm_immediate_exec ) {
        KA_TRACE( 20, ( "__kmp_join_call: T#%d, old team = %p old task_team = %p, th_task_team = %p\n",
                         __kmp_gtid_from_thread( master_th ), team,
                         team -> t.t_task_team, master_th->th.th_task_team) );
        KMP_DEBUG_ASSERT( master_th->th.th_task_team == team->t.t_task_team );
    }
#endif // OMP_30_ENABLED

    if( team->t.t_serialized ) {
#if OMP_40_ENABLED
        if ( master_th->th.th_team_microtask ) {
            // We are in teams construct
            int level = team->t.t_level;
            int tlevel = master_th->th.th_teams_level;
            if ( level == tlevel ) {
                // AC: we haven't incremented it earlier at start of teams construct,
                //     so do it here - at the end of teams construct
                team->t.t_level++;
            } else if ( level == tlevel + 1 ) {
                // AC: we are exiting parallel inside teams, need to increment serialization
                //     in order to restore it in the next call to __kmpc_end_serialized_parallel
                team->t.t_serialized++;
            }
        }
#endif /* OMP_40_ENABLED */
        __kmpc_end_serialized_parallel( loc, gtid );
        return;
    }

    master_active = team->t.t_master_active;

#if OMP_40_ENABLED
    if (!exit_teams)
#endif /* OMP_40_ENABLED */
    {
        // AC: No barrier for internal teams at exit from teams construct.
        //     But there is barrier for external team (league).
        __kmp_internal_join( loc, gtid, team );
    }
    KMP_MB();

#if USE_ITT_BUILD
    if ( __itt_stack_caller_create_ptr ) {
        __kmp_itt_stack_caller_destroy( (__itt_caller)team->t.t_stack_id ); // destroy the stack stitching id after join barrier
    }

    // Mark end of "parallel" region for VTune. Only use one of frame notification scheme at the moment.
    if ( ( __itt_frame_end_v3_ptr && __kmp_forkjoin_frames && ! __kmp_forkjoin_frames_mode ) || KMP_ITT_DEBUG )
# if OMP_40_ENABLED
    if ( !master_th->th.th_team_microtask /* not in teams */ ||
         ( !exit_teams && team->t.t_level == master_th->th.th_teams_level ) )
        // Either not in teams or exiting teams region
        // (teams is a frame and no other frames inside the teams)
# endif /* OMP_40_ENABLED */
    {
        master_th->th.th_ident = loc;
        __kmp_itt_region_joined( gtid );
    }
#endif /* USE_ITT_BUILD */

#if OMP_40_ENABLED
    if ( master_th->th.th_team_microtask &&
         !exit_teams &&
         team->t.t_pkfn != (microtask_t)__kmp_teams_master &&
         team->t.t_level == master_th->th.th_teams_level + 1 ) {
        // AC: We need to leave the team structure intact at the end
        //     of parallel inside the teams construct, so that at the next
        //     parallel same (hot) team works, only adjust nesting levels

        /* Decrement our nested depth level */
        team->t.t_level --;
        team->t.t_active_level --;
        KMP_TEST_THEN_DEC32( (kmp_int32*) &root->r.r_in_parallel );

        /* Restore number of threads in the team if needed */
        if ( master_th->th.th_team_nproc < master_th->th.th_set_nth_teams ) {
            int old_num = master_th->th.th_team_nproc;
            int new_num = master_th->th.th_set_nth_teams;
            kmp_info_t **other_threads = team->t.t_threads;
            team->t.t_nproc = new_num;
            for ( i = 0; i < old_num; ++i ) {
                other_threads[i]->th.th_team_nproc = new_num;
            }
            // Adjust states of non-used threads of the team
            for ( i = old_num; i < new_num; ++i ) {
                // Re-initialize thread's barrier data.
                int b;
                kmp_balign_t * balign = other_threads[i]->th.th_bar;
                for ( b = 0; b < bp_last_bar; ++ b ) {
                    balign[ b ].bb.b_arrived        = team->t.t_bar[ b ].b_arrived;
                }
                // Synchronize thread's task state
                other_threads[i]->th.th_task_state = master_th->th.th_task_state;
            }
        }
        return;
    }
#endif /* OMP_40_ENABLED */
    /* do cleanup and restore the parent team */
    master_th -> th.th_info .ds.ds_tid = team -> t.t_master_tid;
    master_th -> th.th_local.this_construct = team -> t.t_master_this_cons;
    master_th -> th.th_local.last_construct = team -> t.t_master_last_cons;

    master_th -> th.th_dispatch =
                & parent_team -> t.t_dispatch[ team -> t.t_master_tid ];

    /* jc: The following lock has instructions with REL and ACQ semantics,
       separating the parallel user code called in this parallel region
       from the serial user code called after this function returns.
    */
    __kmp_acquire_bootstrap_lock( &__kmp_forkjoin_lock );

#if OMP_40_ENABLED
    if ( !master_th->th.th_team_microtask || team->t.t_level > master_th->th.th_teams_level )
#endif /* OMP_40_ENABLED */
    {
        /* Decrement our nested depth level */
        KMP_TEST_THEN_DEC32( (kmp_int32*) &root->r.r_in_parallel );
    }
    KMP_DEBUG_ASSERT( root->r.r_in_parallel >= 0 );

    #if OMP_30_ENABLED
    KF_TRACE( 10, ("__kmp_join_call1: T#%d, this_thread=%p team=%p\n",
                   0, master_th, team ) );
    __kmp_pop_current_task_from_thread( master_th );
    #endif // OMP_30_ENABLED

#if OMP_40_ENABLED && KMP_AFFINITY_SUPPORTED
    //
    // Restore master thread's partition.
    //
    master_th -> th.th_first_place = team -> t.t_first_place;
    master_th -> th.th_last_place = team -> t.t_last_place;
#endif /* OMP_40_ENABLED */

#if KMP_ARCH_X86 || KMP_ARCH_X86_64
    if ( __kmp_inherit_fp_control && team->t.t_fp_control_saved ) {
        __kmp_clear_x87_fpu_status_word();
        __kmp_load_x87_fpu_control_word( &team->t.t_x87_fpu_control_word );
        __kmp_load_mxcsr( &team->t.t_mxcsr );
    }
#endif /* KMP_ARCH_X86 || KMP_ARCH_X86_64 */

    if ( root -> r.r_active != master_active )
        root -> r.r_active = master_active;

    __kmp_free_team( root, team ); /* this will free worker threads */

    /* this race was fun to find.  make sure the following is in the critical
     * region otherwise assertions may fail occasiounally since the old team
     * may be reallocated and the hierarchy appears inconsistent.  it is
     * actually safe to run and won't cause any bugs, but will cause thoose
     * assertion failures.  it's only one deref&assign so might as well put this
     * in the critical region */
    master_th -> th.th_team        =   parent_team;
    master_th -> th.th_team_nproc  =   parent_team -> t.t_nproc;
    master_th -> th.th_team_master =   parent_team -> t.t_threads[0];
    master_th -> th.th_team_serialized = parent_team -> t.t_serialized;

    /* restore serialized team, if need be */
    if( parent_team -> t.t_serialized &&
        parent_team != master_th->th.th_serial_team &&
        parent_team != root->r.r_root_team ) {
            __kmp_free_team( root, master_th -> th.th_serial_team );
            master_th -> th.th_serial_team = parent_team;
    }

#if OMP_30_ENABLED
    if ( __kmp_tasking_mode != tskm_immediate_exec ) {
        //
        // Copy the task team from the new child / old parent team
        // to the thread.  If non-NULL, copy the state flag also.
        //
        if ( ( master_th -> th.th_task_team = parent_team -> t.t_task_team ) != NULL ) {
            master_th -> th.th_task_state = master_th -> th.th_task_team -> tt.tt_state;
        }
        KA_TRACE( 20, ( "__kmp_join_call: Master T#%d restoring task_team %p / team %p\n",
                        __kmp_gtid_from_thread( master_th ), master_th->th.th_task_team,
                        parent_team ) );
    }
#endif /* OMP_30_ENABLED */

    #if OMP_30_ENABLED
         // TODO: GEH - cannot do this assertion because root thread not set up as executing
         // KMP_ASSERT( master_th->th.th_current_task->td_flags.executing == 0 );
         master_th->th.th_current_task->td_flags.executing = 1;
    #endif // OMP_30_ENABLED

    __kmp_release_bootstrap_lock( &__kmp_forkjoin_lock );

    KMP_MB();
    KA_TRACE( 20, ("__kmp_join_call: exit T#%d\n", gtid ));
}

/* ------------------------------------------------------------------------ */
/* ------------------------------------------------------------------------ */

/* Check whether we should push an internal control record onto the
   serial team stack.  If so, do it.  */
void
__kmp_save_internal_controls ( kmp_info_t * thread )
{

    if ( thread -> th.th_team != thread -> th.th_serial_team ) {
        return;
    }
    if (thread -> th.th_team -> t.t_serialized > 1) {
        int push = 0;

        if (thread -> th.th_team -> t.t_control_stack_top == NULL) {
            push = 1;
        } else {
            if ( thread -> th.th_team -> t.t_control_stack_top -> serial_nesting_level !=
                 thread -> th.th_team -> t.t_serialized ) {
                push = 1;
            }
        }
        if (push) {  /* push a record on the serial team's stack */
            kmp_internal_control_t * control = (kmp_internal_control_t *) __kmp_allocate(sizeof(kmp_internal_control_t));

#if OMP_30_ENABLED
            copy_icvs( control, & thread->th.th_current_task->td_icvs );
#else
            control->nproc        = thread->th.th_team->t.t_set_nproc[0];
            control->dynamic      = thread->th.th_team->t.t_set_dynamic[0];
            control->nested       = thread->th.th_team->t.t_set_nested[0];
            control->blocktime    = thread->th.th_team->t.t_set_blocktime[0];
            control->bt_intervals = thread->th.th_team->t.t_set_bt_intervals[0];
            control->bt_set       = thread->th.th_team->t.t_set_bt_set[0];
#endif // OMP_30_ENABLED

            control->serial_nesting_level = thread->th.th_team->t.t_serialized;

            control->next = thread -> th.th_team -> t.t_control_stack_top;
            thread -> th.th_team -> t.t_control_stack_top = control;
        }
    }
}

/* Changes set_nproc */
void
__kmp_set_num_threads( int new_nth, int gtid )
{
    kmp_info_t *thread;
    kmp_root_t *root;

    KF_TRACE( 10, ("__kmp_set_num_threads: new __kmp_nth = %d\n", new_nth ));
    KMP_DEBUG_ASSERT( __kmp_init_serial );

    if (new_nth < 1)
        new_nth = 1;
    else if (new_nth > __kmp_max_nth)
        new_nth = __kmp_max_nth;

    thread = __kmp_threads[gtid];

    __kmp_save_internal_controls( thread );

    set__nproc( thread, new_nth );

    //
    // If this omp_set_num_threads() call will cause the hot team size to be
    // reduced (in the absence of a num_threads clause), then reduce it now,
    // rather than waiting for the next parallel region.
    //
    root = thread->th.th_root;
    if ( __kmp_init_parallel && ( ! root->r.r_active )
      && ( root->r.r_hot_team->t.t_nproc > new_nth ) ) {
        kmp_team_t *hot_team = root->r.r_hot_team;
        int f;

        __kmp_acquire_bootstrap_lock( &__kmp_forkjoin_lock );


#if OMP_30_ENABLED
        if ( __kmp_tasking_mode != tskm_immediate_exec ) {
            kmp_task_team_t *task_team = hot_team->t.t_task_team;
            if ( ( task_team != NULL ) && TCR_SYNC_4(task_team->tt.tt_active) ) {
                //
                // Signal the worker threads (esp. the extra ones) to stop
                // looking for tasks while spin waiting.  The task teams
                // are reference counted and will be deallocated by the
                // last worker thread.
                //
                KMP_DEBUG_ASSERT( hot_team->t.t_nproc > 1 );
                TCW_SYNC_4( task_team->tt.tt_active, FALSE );
                KMP_MB();

                KA_TRACE( 20, ( "__kmp_set_num_threads: setting task_team %p to NULL\n",
                  &hot_team->t.t_task_team ) );
                  hot_team->t.t_task_team = NULL;
            }
            else {
                KMP_DEBUG_ASSERT( task_team == NULL );
            }
        }
#endif // OMP_30_ENABLED

        //
        // Release the extra threads we don't need any more.
        //
        for ( f = new_nth;  f < hot_team->t.t_nproc; f++ ) {
            KMP_DEBUG_ASSERT( hot_team->t.t_threads[f] != NULL );
            __kmp_free_thread( hot_team->t.t_threads[f] );
            hot_team->t.t_threads[f] =  NULL;
        }
        hot_team->t.t_nproc = new_nth;


        __kmp_release_bootstrap_lock( &__kmp_forkjoin_lock );

        //
        // Update the t_nproc field in the threads that are still active.
        //
        for( f=0 ; f < new_nth; f++ ) {
            KMP_DEBUG_ASSERT( hot_team->t.t_threads[f] != NULL );
            hot_team->t.t_threads[f]->th.th_team_nproc = new_nth;
        }
#if KMP_MIC
        // Special flag in case omp_set_num_threads() call
        hot_team -> t.t_size_changed = -1;
#endif
    }

}

#if OMP_30_ENABLED
/* Changes max_active_levels */
void
__kmp_set_max_active_levels( int gtid, int max_active_levels )
{
    kmp_info_t *thread;

    KF_TRACE( 10, ( "__kmp_set_max_active_levels: new max_active_levels for thread %d = (%d)\n", gtid, max_active_levels ) );
    KMP_DEBUG_ASSERT( __kmp_init_serial );

    // validate max_active_levels
    if( max_active_levels < 0 ) {
        KMP_WARNING( ActiveLevelsNegative, max_active_levels );
        // We ignore this call if the user has specified a negative value.
        // The current setting won't be changed. The last valid setting will be used.
        // A warning will be issued (if warnings are allowed as controlled by the KMP_WARNINGS env var).
        KF_TRACE( 10, ( "__kmp_set_max_active_levels: the call is ignored: new max_active_levels for thread %d = (%d)\n", gtid, max_active_levels ) );
        return;
    }
    if( max_active_levels <= KMP_MAX_ACTIVE_LEVELS_LIMIT ) {
        // it's OK, the max_active_levels is within the valid range: [ 0; KMP_MAX_ACTIVE_LEVELS_LIMIT ]
        // We allow a zero value. (implementation defined behavior)
    } else {
        KMP_WARNING( ActiveLevelsExceedLimit, max_active_levels, KMP_MAX_ACTIVE_LEVELS_LIMIT  );
        max_active_levels = KMP_MAX_ACTIVE_LEVELS_LIMIT;
        // Current upper limit is MAX_INT. (implementation defined behavior)
        // If the input exceeds the upper limit, we correct the input to be the upper limit. (implementation defined behavior)
        // Actually, the flow should never get here until we use MAX_INT limit.
    }
    KF_TRACE( 10, ( "__kmp_set_max_active_levels: after validation: new max_active_levels for thread %d = (%d)\n", gtid, max_active_levels ) );

    thread = __kmp_threads[ gtid ];

    __kmp_save_internal_controls( thread );

    set__max_active_levels( thread, max_active_levels );

}

/* Gets max_active_levels */
int
__kmp_get_max_active_levels( int gtid )
{
    kmp_info_t *thread;

    KF_TRACE( 10, ( "__kmp_get_max_active_levels: thread %d\n", gtid ) );
    KMP_DEBUG_ASSERT( __kmp_init_serial );

    thread = __kmp_threads[ gtid ];
    KMP_DEBUG_ASSERT( thread -> th.th_current_task );
    KF_TRACE( 10, ( "__kmp_get_max_active_levels: thread %d, curtask=%p, curtask_maxaclevel=%d\n",
        gtid, thread -> th.th_current_task, thread -> th.th_current_task -> td_icvs.max_active_levels ) );
    return thread -> th.th_current_task -> td_icvs.max_active_levels;
}

/* Changes def_sched_var ICV values (run-time schedule kind and chunk) */
void
__kmp_set_schedule( int gtid, kmp_sched_t kind, int chunk )
{
    kmp_info_t *thread;
//    kmp_team_t *team;

    KF_TRACE( 10, ("__kmp_set_schedule: new schedule for thread %d = (%d, %d)\n", gtid, (int)kind, chunk ));
    KMP_DEBUG_ASSERT( __kmp_init_serial );

    // Check if the kind parameter is valid, correct if needed.
    // Valid parameters should fit in one of two intervals - standard or extended:
    //       <lower>, <valid>, <upper_std>, <lower_ext>, <valid>, <upper>
    // 2008-01-25: 0,  1 - 4,       5,         100,     101 - 102, 103
    if ( kind <= kmp_sched_lower || kind >= kmp_sched_upper ||
       ( kind <= kmp_sched_lower_ext && kind >= kmp_sched_upper_std ) )
    {
        // TODO: Hint needs attention in case we change the default schedule.
        __kmp_msg(
            kmp_ms_warning,
            KMP_MSG( ScheduleKindOutOfRange, kind ),
            KMP_HNT( DefaultScheduleKindUsed, "static, no chunk" ),
            __kmp_msg_null
        );
        kind = kmp_sched_default;
        chunk = 0;         // ignore chunk value in case of bad kind
    }

    thread = __kmp_threads[ gtid ];

    __kmp_save_internal_controls( thread );

    if ( kind < kmp_sched_upper_std ) {
        if ( kind == kmp_sched_static && chunk < KMP_DEFAULT_CHUNK ) {
            // differ static chunked vs. unchunked:
            // chunk should be invalid to indicate unchunked schedule (which is the default)
            thread -> th.th_current_task -> td_icvs.sched.r_sched_type = kmp_sch_static;
        } else {
            thread -> th.th_current_task -> td_icvs.sched.r_sched_type = __kmp_sch_map[ kind - kmp_sched_lower - 1 ];
        }
    } else {
        //    __kmp_sch_map[ kind - kmp_sched_lower_ext + kmp_sched_upper_std - kmp_sched_lower - 2 ];
        thread -> th.th_current_task -> td_icvs.sched.r_sched_type =
            __kmp_sch_map[ kind - kmp_sched_lower_ext + kmp_sched_upper_std - kmp_sched_lower - 2 ];
    }
    if ( kind == kmp_sched_auto ) {
        // ignore parameter chunk for schedule auto
        thread -> th.th_current_task -> td_icvs.sched.chunk = KMP_DEFAULT_CHUNK;
    } else {
        thread -> th.th_current_task -> td_icvs.sched.chunk = chunk;
    }
}

/* Gets def_sched_var ICV values */
void
__kmp_get_schedule( int gtid, kmp_sched_t * kind, int * chunk )
{
    kmp_info_t     *thread;
    enum sched_type th_type;
    int             i;

    KF_TRACE( 10, ("__kmp_get_schedule: thread %d\n", gtid ));
    KMP_DEBUG_ASSERT( __kmp_init_serial );

    thread = __kmp_threads[ gtid ];

    //th_type = thread -> th.th_team -> t.t_set_sched[ thread->th.th_info.ds.ds_tid ].r_sched_type;
    th_type = thread -> th.th_current_task -> td_icvs.sched.r_sched_type;

    switch ( th_type ) {
    case kmp_sch_static:
    case kmp_sch_static_greedy:
    case kmp_sch_static_balanced:
        *kind = kmp_sched_static;
        *chunk = 0;   // chunk was not set, try to show this fact via zero value
        return;
    case kmp_sch_static_chunked:
        *kind = kmp_sched_static;
        break;
    case kmp_sch_dynamic_chunked:
        *kind = kmp_sched_dynamic;
        break;
    case kmp_sch_guided_chunked:
    case kmp_sch_guided_iterative_chunked:
    case kmp_sch_guided_analytical_chunked:
        *kind = kmp_sched_guided;
        break;
    case kmp_sch_auto:
        *kind = kmp_sched_auto;
        break;
    case kmp_sch_trapezoidal:
        *kind = kmp_sched_trapezoidal;
        break;
/*
    case kmp_sch_static_steal:
        *kind = kmp_sched_static_steal;
        break;
*/
    default:
        KMP_FATAL( UnknownSchedulingType, th_type );
    }

    //*chunk = thread -> th.th_team -> t.t_set_sched[ thread->th.th_info.ds.ds_tid ].chunk;
    *chunk = thread -> th.th_current_task -> td_icvs.sched.chunk;
}

int
__kmp_get_ancestor_thread_num( int gtid, int level ) {

    int ii, dd;
    kmp_team_t *team;
    kmp_info_t *thr;

    KF_TRACE( 10, ("__kmp_get_ancestor_thread_num: thread %d %d\n", gtid, level ));
    KMP_DEBUG_ASSERT( __kmp_init_serial );

    // validate level
    if( level == 0 ) return 0;
    if( level < 0 ) return -1;
    thr = __kmp_threads[ gtid ];
    team = thr->th.th_team;
    ii = team -> t.t_level;
    if( level > ii ) return -1;

#if OMP_40_ENABLED
    if( thr->th.th_team_microtask ) {
        // AC: we are in teams region where multiple nested teams have same level
        int tlevel = thr->th.th_teams_level; // the level of the teams construct
        if( level <= tlevel ) { // otherwise usual algorithm works (will not touch the teams)
            KMP_DEBUG_ASSERT( ii >= tlevel );
            // AC: As we need to pass by the teams league, we need to artificially increase ii
            if ( ii == tlevel ) {
                ii += 2; // three teams have same level
            } else {
                ii ++;   // two teams have same level
            }
        }
    }
#endif

    if( ii == level ) return __kmp_tid_from_gtid( gtid );

    dd = team -> t.t_serialized;
    level++;
    while( ii > level )
    {
        for( dd = team -> t.t_serialized; ( dd > 0 ) && ( ii > level ); dd--, ii-- )
        {
        }
        if( ( team -> t.t_serialized ) && ( !dd ) ) {
            team = team->t.t_parent;
            continue;
        }
        if( ii > level ) {
            team = team->t.t_parent;
            dd = team -> t.t_serialized;
            ii--;
        }
    }

    return ( dd > 1 ) ? ( 0 ) : ( team -> t.t_master_tid );
}

int
__kmp_get_team_size( int gtid, int level ) {

    int ii, dd;
    kmp_team_t *team;
    kmp_info_t *thr;

    KF_TRACE( 10, ("__kmp_get_team_size: thread %d %d\n", gtid, level ));
    KMP_DEBUG_ASSERT( __kmp_init_serial );

    // validate level
    if( level == 0 ) return 1;
    if( level < 0 ) return -1;
    thr = __kmp_threads[ gtid ];
    team = thr->th.th_team;
    ii = team -> t.t_level;
    if( level > ii ) return -1;

#if OMP_40_ENABLED
    if( thr->th.th_team_microtask ) {
        // AC: we are in teams region where multiple nested teams have same level
        int tlevel = thr->th.th_teams_level; // the level of the teams construct
        if( level <= tlevel ) { // otherwise usual algorithm works (will not touch the teams)
            KMP_DEBUG_ASSERT( ii >= tlevel );
            // AC: As we need to pass by the teams league, we need to artificially increase ii
            if ( ii == tlevel ) {
                ii += 2; // three teams have same level
            } else {
                ii ++;   // two teams have same level
            }
        }
    }
#endif

    while( ii > level )
    {
        for( dd = team -> t.t_serialized; ( dd > 0 ) && ( ii > level ); dd--, ii-- )
        {
        }
        if( team -> t.t_serialized && ( !dd ) ) {
            team = team->t.t_parent;
            continue;
        }
        if( ii > level ) {
            team = team->t.t_parent;
            ii--;
        }
    }

    return team -> t.t_nproc;
}

#endif // OMP_30_ENABLED

kmp_r_sched_t
__kmp_get_schedule_global() {
// This routine created because pairs (__kmp_sched, __kmp_chunk) and (__kmp_static, __kmp_guided)
// may be changed by kmp_set_defaults independently. So one can get the updated schedule here.

    kmp_r_sched_t r_sched;

    // create schedule from 4 globals: __kmp_sched, __kmp_chunk, __kmp_static, __kmp_guided
    // __kmp_sched should keep original value, so that user can set KMP_SCHEDULE multiple times,
    // and thus have different run-time schedules in different roots (even in OMP 2.5)
    if ( __kmp_sched == kmp_sch_static ) {
        r_sched.r_sched_type = __kmp_static; // replace STATIC with more detailed schedule (balanced or greedy)
    } else if ( __kmp_sched == kmp_sch_guided_chunked ) {
        r_sched.r_sched_type = __kmp_guided; // replace GUIDED with more detailed schedule (iterative or analytical)
    } else {
        r_sched.r_sched_type = __kmp_sched;  // (STATIC_CHUNKED), or (DYNAMIC_CHUNKED), or other
    }

    if ( __kmp_chunk < KMP_DEFAULT_CHUNK ) { // __kmp_chunk may be wrong here (if it was not ever set)
        r_sched.chunk = KMP_DEFAULT_CHUNK;
    } else {
        r_sched.chunk = __kmp_chunk;
    }

    return r_sched;
}

/* ------------------------------------------------------------------------ */
/* ------------------------------------------------------------------------ */


/*
 * Allocate (realloc == FALSE) * or reallocate (realloc == TRUE)
 * at least argc number of *t_argv entries for the requested team.
 */
static void
__kmp_alloc_argv_entries( int argc, kmp_team_t *team, int realloc )
{

    KMP_DEBUG_ASSERT( team );
    if( !realloc || argc > team -> t.t_max_argc ) {

        KA_TRACE( 100, ( "__kmp_alloc_argv_entries: team %d: needed entries=%d, current entries=%d\n",
                         team->t.t_id, argc, ( realloc ) ? team->t.t_max_argc : 0 ));
#if (KMP_PERF_V106 == KMP_ON)
        /* if previously allocated heap space for args, free them */
        if ( realloc && team -> t.t_argv != &team -> t.t_inline_argv[0] )
            __kmp_free( (void *) team -> t.t_argv );

        if ( argc <= KMP_INLINE_ARGV_ENTRIES ) {
            /* use unused space in the cache line for arguments */
            team -> t.t_max_argc = KMP_INLINE_ARGV_ENTRIES;
            KA_TRACE( 100, ( "__kmp_alloc_argv_entries: team %d: inline allocate %d argv entries\n",
                             team->t.t_id, team->t.t_max_argc ));
            team -> t.t_argv = &team -> t.t_inline_argv[0];
            if ( __kmp_storage_map ) {
                __kmp_print_storage_map_gtid( -1, &team->t.t_inline_argv[0],
                                         &team->t.t_inline_argv[KMP_INLINE_ARGV_ENTRIES],
                                         (sizeof(void *) * KMP_INLINE_ARGV_ENTRIES),
                                         "team_%d.t_inline_argv",
                                         team->t.t_id );
            }
        } else {
            /* allocate space for arguments in the heap */
            team -> t.t_max_argc = ( argc <= (KMP_MIN_MALLOC_ARGV_ENTRIES >> 1 )) ?
                                     KMP_MIN_MALLOC_ARGV_ENTRIES : 2 * argc;
            KA_TRACE( 100, ( "__kmp_alloc_argv_entries: team %d: dynamic allocate %d argv entries\n",
                             team->t.t_id, team->t.t_max_argc ));
            team -> t.t_argv     = (void**) __kmp_page_allocate( sizeof(void*) * team->t.t_max_argc );
            if ( __kmp_storage_map ) {
                __kmp_print_storage_map_gtid( -1, &team->t.t_argv[0], &team->t.t_argv[team->t.t_max_argc],
                                         sizeof(void *) * team->t.t_max_argc, "team_%d.t_argv",
                                         team->t.t_id );
            }
        }
#else /* KMP_PERF_V106 == KMP_OFF */
        if ( realloc )
            __kmp_free( (void*) team -> t.t_argv );
        team -> t.t_max_argc = ( argc <= (KMP_MIN_MALLOC_ARGV_ENTRIES >> 1 )) ?
                             KMP_MIN_MALLOC_ARGV_ENTRIES : 2 * argc;
        KA_TRACE( 100, ( "__kmp_alloc_argv_entries: team %d: dynamic allocate %d argv entries\n",
                         team->t.t_id, team->t.t_max_argc ));
        team -> t.t_argv     = __kmp_page_allocate( sizeof(void*) * team->t.t_max_argc );
        if ( __kmp_storage_map ) {
            __kmp_print_storage_map_gtid( -1, &team->t.t_argv[0], &team->t.t_argv[team->t.t_max_argc],
                                     sizeof(void *) * team->t.t_max_argc, "team_%d.t_argv", team->t.t_id );
        }
#endif /* KMP_PERF_V106 */

    }
}

static void
__kmp_allocate_team_arrays(kmp_team_t *team, int max_nth)
{
    int i;
    int num_disp_buff = max_nth > 1 ? KMP_MAX_DISP_BUF : 2;
#if KMP_USE_POOLED_ALLOC
    // AC: TODO: fix bug here: size of t_disp_buffer should not be multiplied by max_nth!
    char *ptr = __kmp_allocate(max_nth *
                            ( sizeof(kmp_info_t*) + sizeof(dispatch_shared_info_t)*num_disp_buf
                               + sizeof(kmp_disp_t) + sizeof(int)*6
#  if OMP_30_ENABLED
                               //+ sizeof(int)
                               + sizeof(kmp_r_sched_t)
                               + sizeof(kmp_taskdata_t)
#  endif // OMP_30_ENABLED
                        )     );

    team -> t.t_threads          = (kmp_info_t**) ptr; ptr += sizeof(kmp_info_t*) * max_nth;
    team -> t.t_disp_buffer      = (dispatch_shared_info_t*) ptr;
                                   ptr += sizeof(dispatch_shared_info_t) * num_disp_buff;
    team -> t.t_dispatch         = (kmp_disp_t*) ptr; ptr += sizeof(kmp_disp_t) * max_nth;
    team -> t.t_set_nproc        = (int*) ptr; ptr += sizeof(int) * max_nth;
    team -> t.t_set_dynamic      = (int*) ptr; ptr += sizeof(int) * max_nth;
    team -> t.t_set_nested       = (int*) ptr; ptr += sizeof(int) * max_nth;
    team -> t.t_set_blocktime    = (int*) ptr; ptr += sizeof(int) * max_nth;
    team -> t.t_set_bt_intervals = (int*) ptr; ptr += sizeof(int) * max_nth;
    team -> t.t_set_bt_set       = (int*) ptr;
#  if OMP_30_ENABLED
    ptr += sizeof(int) * max_nth;
    //team -> t.t_set_max_active_levels = (int*) ptr; ptr += sizeof(int) * max_nth;
    team -> t.t_set_sched        = (kmp_r_sched_t*) ptr;
    ptr += sizeof(kmp_r_sched_t) * max_nth;
    team -> t.t_implicit_task_taskdata = (kmp_taskdata_t*) ptr;
    ptr += sizeof(kmp_taskdata_t) * max_nth;
#  endif // OMP_30_ENABLED
#else

    team -> t.t_threads = (kmp_info_t**) __kmp_allocate( sizeof(kmp_info_t*) * max_nth );
    team -> t.t_disp_buffer = (dispatch_shared_info_t*)
        __kmp_allocate( sizeof(dispatch_shared_info_t) * num_disp_buff );
    team -> t.t_dispatch = (kmp_disp_t*) __kmp_allocate( sizeof(kmp_disp_t) * max_nth );
    #if OMP_30_ENABLED
    //team -> t.t_set_max_active_levels = (int*) __kmp_allocate( sizeof(int) * max_nth );
    //team -> t.t_set_sched = (kmp_r_sched_t*) __kmp_allocate( sizeof(kmp_r_sched_t) * max_nth );
    team -> t.t_implicit_task_taskdata = (kmp_taskdata_t*) __kmp_allocate( sizeof(kmp_taskdata_t) * max_nth );
    #else
    team -> t.t_set_nproc = (int*) __kmp_allocate( sizeof(int) * max_nth );
    team -> t.t_set_dynamic = (int*) __kmp_allocate( sizeof(int) * max_nth );
    team -> t.t_set_nested = (int*) __kmp_allocate( sizeof(int) * max_nth );
    team -> t.t_set_blocktime = (int*) __kmp_allocate( sizeof(int) * max_nth );
    team -> t.t_set_bt_intervals = (int*) __kmp_allocate( sizeof(int) * max_nth );
    team -> t.t_set_bt_set = (int*) __kmp_allocate( sizeof(int) * max_nth );
#  endif // OMP_30_ENABLED
#endif
    team->t.t_max_nproc = max_nth;

    /* setup dispatch buffers */
    for(i = 0 ; i < num_disp_buff; ++i)
        team -> t.t_disp_buffer[i].buffer_index = i;
}

static void
__kmp_free_team_arrays(kmp_team_t *team) {
    /* Note: this does not free the threads in t_threads (__kmp_free_threads) */
    int i;
    for ( i = 0; i < team->t.t_max_nproc; ++ i ) {
        if ( team->t.t_dispatch[ i ].th_disp_buffer != NULL ) {
            __kmp_free( team->t.t_dispatch[ i ].th_disp_buffer );
            team->t.t_dispatch[ i ].th_disp_buffer = NULL;
        }; // if
    }; // for
    __kmp_free(team->t.t_threads);
    #if !KMP_USE_POOLED_ALLOC
        __kmp_free(team->t.t_disp_buffer);
        __kmp_free(team->t.t_dispatch);
        #if OMP_30_ENABLED
        //__kmp_free(team->t.t_set_max_active_levels);
        //__kmp_free(team->t.t_set_sched);
        __kmp_free(team->t.t_implicit_task_taskdata);
        #else
        __kmp_free(team->t.t_set_nproc);
        __kmp_free(team->t.t_set_dynamic);
        __kmp_free(team->t.t_set_nested);
        __kmp_free(team->t.t_set_blocktime);
        __kmp_free(team->t.t_set_bt_intervals);
        __kmp_free(team->t.t_set_bt_set);
    #  endif // OMP_30_ENABLED
    #endif
    team->t.t_threads     = NULL;
    team->t.t_disp_buffer = NULL;
    team->t.t_dispatch    = NULL;
#if OMP_30_ENABLED
    //team->t.t_set_sched   = 0;
    //team->t.t_set_max_active_levels = 0;
    team->t.t_implicit_task_taskdata = 0;
#else
    team->t.t_set_nproc   = 0;
    team->t.t_set_dynamic = 0;
    team->t.t_set_nested  = 0;
    team->t.t_set_blocktime   = 0;
    team->t.t_set_bt_intervals = 0;
    team->t.t_set_bt_set  = 0;
#endif // OMP_30_ENABLED
}

static void
__kmp_reallocate_team_arrays(kmp_team_t *team, int max_nth) {
    kmp_info_t **oldThreads = team->t.t_threads;

    #if !KMP_USE_POOLED_ALLOC
        __kmp_free(team->t.t_disp_buffer);
        __kmp_free(team->t.t_dispatch);
        #if OMP_30_ENABLED
        //__kmp_free(team->t.t_set_max_active_levels);
        //__kmp_free(team->t.t_set_sched);
        __kmp_free(team->t.t_implicit_task_taskdata);
        #else
        __kmp_free(team->t.t_set_nproc);
        __kmp_free(team->t.t_set_dynamic);
        __kmp_free(team->t.t_set_nested);
        __kmp_free(team->t.t_set_blocktime);
        __kmp_free(team->t.t_set_bt_intervals);
        __kmp_free(team->t.t_set_bt_set);
    #  endif // OMP_30_ENABLED
    #endif
    __kmp_allocate_team_arrays(team, max_nth);

    memcpy(team->t.t_threads, oldThreads, team->t.t_nproc * sizeof (kmp_info_t*));

    __kmp_free(oldThreads);
}

static kmp_internal_control_t
__kmp_get_global_icvs( void ) {

#if OMP_30_ENABLED
    kmp_r_sched_t r_sched = __kmp_get_schedule_global(); // get current state of scheduling globals
#endif /* OMP_30_ENABLED */

#if OMP_40_ENABLED
    KMP_DEBUG_ASSERT( __kmp_nested_proc_bind.used > 0 );
#endif /* OMP_40_ENABLED */

    kmp_internal_control_t g_icvs = {
      0,                            //int serial_nesting_level; //corresponds to the value of the th_team_serialized field
      __kmp_dflt_nested,            //int nested;               //internal control for nested parallelism (per thread)
      __kmp_global.g.g_dynamic,                                 //internal control for dynamic adjustment of threads (per thread)
      __kmp_dflt_team_nth,
                                    //int nproc;                //internal control for # of threads for next parallel region (per thread)
                                    // (use a max ub on value if __kmp_parallel_initialize not called yet)
      __kmp_dflt_blocktime,         //int blocktime;            //internal control for blocktime
      __kmp_bt_intervals,           //int bt_intervals;         //internal control for blocktime intervals
      __kmp_env_blocktime,          //int bt_set;               //internal control for whether blocktime is explicitly set
#if OMP_30_ENABLED
      __kmp_dflt_max_active_levels, //int max_active_levels;    //internal control for max_active_levels
      r_sched,                      //kmp_r_sched_t sched;      //internal control for runtime schedule {sched,chunk} pair
#endif /* OMP_30_ENABLED */
#if OMP_40_ENABLED
      __kmp_nested_proc_bind.bind_types[0],
#endif /* OMP_40_ENABLED */
      NULL                          //struct kmp_internal_control *next;
    };

    return g_icvs;
}

static kmp_internal_control_t
__kmp_get_x_global_icvs( const kmp_team_t *team ) {

    #if OMP_30_ENABLED
    kmp_internal_control_t gx_icvs;
    gx_icvs.serial_nesting_level = 0; // probably =team->t.t_serial like in save_inter_controls
    copy_icvs( & gx_icvs, & team->t.t_threads[0]->th.th_current_task->td_icvs );
    gx_icvs.next = NULL;
    #else
    kmp_internal_control_t gx_icvs =
    {
      0,
      team->t.t_set_nested[0],
      team->t.t_set_dynamic[0],
      team->t.t_set_nproc[0],
      team->t.t_set_blocktime[0],
      team->t.t_set_bt_intervals[0],
      team->t.t_set_bt_set[0],
      NULL                          //struct kmp_internal_control *next;
    };
    #endif // OMP_30_ENABLED

    return gx_icvs;
}

static void
__kmp_initialize_root( kmp_root_t *root )
{
    int           f;
    kmp_team_t   *root_team;
    kmp_team_t   *hot_team;
    size_t        disp_size, dispatch_size, bar_size;
    int           hot_team_max_nth;
#if OMP_30_ENABLED
    kmp_r_sched_t r_sched = __kmp_get_schedule_global(); // get current state of scheduling globals
    kmp_internal_control_t r_icvs = __kmp_get_global_icvs();
#endif // OMP_30_ENABLED
    KMP_DEBUG_ASSERT( root );
    KMP_ASSERT( ! root->r.r_begin );

    /* setup the root state structure */
    __kmp_init_lock( &root->r.r_begin_lock );
    root -> r.r_begin        = FALSE;
    root -> r.r_active       = FALSE;
    root -> r.r_in_parallel  = 0;
    root -> r.r_blocktime    = __kmp_dflt_blocktime;
    root -> r.r_nested       = __kmp_dflt_nested;

    /* setup the root team for this task */
    /* allocate the root team structure */
    KF_TRACE( 10, ( "__kmp_initialize_root: before root_team\n" ) );
    root_team =
        __kmp_allocate_team(
            root,
            1,                                                         // new_nproc
            1,                                                         // max_nproc
#if OMP_40_ENABLED
            __kmp_nested_proc_bind.bind_types[0],
#endif
#if OMP_30_ENABLED
            &r_icvs,
#else
            __kmp_dflt_team_nth_ub,                                    // num_treads
            __kmp_global.g.g_dynamic,                                  // dynamic
            __kmp_dflt_nested,                                         // nested
            __kmp_dflt_blocktime,                                      // blocktime
            __kmp_bt_intervals,                                        // bt_intervals
            __kmp_env_blocktime,                                       // bt_set
#endif // OMP_30_ENABLED
            0                                                          // argc
        );

    KF_TRACE( 10, ( "__kmp_initialize_root: after root_team = %p\n", root_team ) );

    root -> r.r_root_team = root_team;
    root_team -> t.t_control_stack_top = NULL;

    /* initialize root team */
    root_team -> t.t_threads[0] = NULL;
    root_team -> t.t_nproc      = 1;
    root_team -> t.t_serialized = 1;
#if OMP_30_ENABLED
    // TODO???: root_team -> t.t_max_active_levels = __kmp_dflt_max_active_levels;
    root_team -> t.t_sched.r_sched_type = r_sched.r_sched_type;
    root_team -> t.t_sched.chunk        = r_sched.chunk;
#endif // OMP_30_ENABLED
    KA_TRACE( 20, ("__kmp_initialize_root: init root team %d arrived: join=%u, plain=%u\n",
                    root_team->t.t_id, KMP_INIT_BARRIER_STATE, KMP_INIT_BARRIER_STATE ));

    /* setup the  hot team for this task */
    /* allocate the hot team structure */
    KF_TRACE( 10, ( "__kmp_initialize_root: before hot_team\n" ) );
    hot_team =
        __kmp_allocate_team(
            root,
            1,                                                         // new_nproc
            __kmp_dflt_team_nth_ub * 2,                                // max_nproc
#if OMP_40_ENABLED
            __kmp_nested_proc_bind.bind_types[0],
#endif
#if OMP_30_ENABLED
            &r_icvs,
#else
            __kmp_dflt_team_nth_ub,                                    // num_treads
            __kmp_global.g.g_dynamic,                                  // dynamic
            __kmp_dflt_nested,                                         // nested
            __kmp_dflt_blocktime,                                      // blocktime
            __kmp_bt_intervals,                                        // bt_intervals
            __kmp_env_blocktime,                                       // bt_set
#endif // OMP_30_ENABLED
            0                                                          // argc
        );
    KF_TRACE( 10, ( "__kmp_initialize_root: after hot_team = %p\n", hot_team ) );

    root -> r.r_hot_team = hot_team;
    root_team -> t.t_control_stack_top = NULL;

    /* first-time initialization */
    hot_team -> t.t_parent = root_team;

    /* initialize hot team */
    hot_team_max_nth = hot_team->t.t_max_nproc;
    for ( f = 0; f < hot_team_max_nth; ++ f ) {
        hot_team -> t.t_threads[ f ] = NULL;
    }; // for
    hot_team -> t.t_nproc = 1;
#if OMP_30_ENABLED
    // TODO???: hot_team -> t.t_max_active_levels = __kmp_dflt_max_active_levels;
    hot_team -> t.t_sched.r_sched_type = r_sched.r_sched_type;
    hot_team -> t.t_sched.chunk        = r_sched.chunk;
#endif // OMP_30_ENABLED
#if KMP_MIC
    hot_team -> t.t_size_changed = 0;
#endif

}

#ifdef KMP_DEBUG


typedef struct kmp_team_list_item {
    kmp_team_p const *           entry;
    struct kmp_team_list_item *  next;
} kmp_team_list_item_t;
typedef kmp_team_list_item_t * kmp_team_list_t;


static void
__kmp_print_structure_team_accum(    // Add team to list of teams.
    kmp_team_list_t     list,        // List of teams.
    kmp_team_p const *  team         // Team to add.
) {

    // List must terminate with item where both entry and next are NULL.
    // Team is added to the list only once.
    // List is sorted in ascending order by team id.
    // Team id is *not* a key.

    kmp_team_list_t l;

    KMP_DEBUG_ASSERT( list != NULL );
    if ( team == NULL ) {
        return;
    }; // if

    __kmp_print_structure_team_accum( list, team->t.t_parent );
    __kmp_print_structure_team_accum( list, team->t.t_next_pool );

    // Search list for the team.
    l = list;
    while ( l->next != NULL && l->entry != team ) {
        l = l->next;
    }; // while
    if ( l->next != NULL ) {
        return;  // Team has been added before, exit.
    }; // if

    // Team is not found. Search list again for insertion point.
    l = list;
    while ( l->next != NULL && l->entry->t.t_id <= team->t.t_id ) {
        l = l->next;
    }; // while

    // Insert team.
    {
        kmp_team_list_item_t * item =
            (kmp_team_list_item_t *)KMP_INTERNAL_MALLOC( sizeof(  kmp_team_list_item_t ) );
        * item = * l;
        l->entry = team;
        l->next  = item;
    }

}

static void
__kmp_print_structure_team(
    char const *       title,
    kmp_team_p const * team

) {
    __kmp_printf( "%s", title );
    if ( team != NULL ) {
        __kmp_printf( "%2x %p\n", team->t.t_id, team );
    } else {
        __kmp_printf( " - (nil)\n" );
    }; // if
}

static void
__kmp_print_structure_thread(
    char const *       title,
    kmp_info_p const * thread

) {
    __kmp_printf( "%s", title );
    if ( thread != NULL ) {
        __kmp_printf( "%2d %p\n", thread->th.th_info.ds.ds_gtid, thread );
    } else {
        __kmp_printf( " - (nil)\n" );
    }; // if
}

static void
__kmp_print_structure(
    void
) {

    kmp_team_list_t list;

    // Initialize list of teams.
    list = (kmp_team_list_item_t *)KMP_INTERNAL_MALLOC( sizeof( kmp_team_list_item_t ) );
    list->entry = NULL;
    list->next  = NULL;

    __kmp_printf( "\n------------------------------\nGlobal Thread Table\n------------------------------\n" );
    {
        int gtid;
        for ( gtid = 0; gtid < __kmp_threads_capacity; ++ gtid ) {
            __kmp_printf( "%2d", gtid );
            if ( __kmp_threads != NULL ) {
                __kmp_printf( " %p", __kmp_threads[ gtid ] );
            }; // if
            if ( __kmp_root != NULL ) {
                __kmp_printf( " %p", __kmp_root[ gtid ] );
            }; // if
            __kmp_printf( "\n" );
        }; // for gtid
    }

    // Print out __kmp_threads array.
    __kmp_printf( "\n------------------------------\nThreads\n------------------------------\n" );
    if ( __kmp_threads != NULL ) {
        int gtid;
        for ( gtid = 0; gtid < __kmp_threads_capacity; ++ gtid ) {
            kmp_info_t const * thread = __kmp_threads[ gtid ];
            if ( thread != NULL ) {
                __kmp_printf( "GTID %2d %p:\n", gtid, thread );
                __kmp_printf(                 "    Our Root:        %p\n", thread->th.th_root );
                __kmp_print_structure_team(   "    Our Team:     ",        thread->th.th_team );
                __kmp_print_structure_team(   "    Serial Team:  ",        thread->th.th_serial_team );
                __kmp_printf(                 "    Threads:      %2d\n",   thread->th.th_team_nproc );
                __kmp_print_structure_thread( "    Master:       ",        thread->th.th_team_master );
                __kmp_printf(                 "    Serialized?:  %2d\n",   thread->th.th_team_serialized );
                __kmp_printf(                 "    Set NProc:    %2d\n",   thread->th.th_set_nproc );
#if OMP_40_ENABLED
                __kmp_printf(                 "    Set Proc Bind: %2d\n",  thread->th.th_set_proc_bind );
#endif
                __kmp_print_structure_thread( "    Next in pool: ",        thread->th.th_next_pool );
                __kmp_printf( "\n" );
                __kmp_print_structure_team_accum( list, thread->th.th_team );
                __kmp_print_structure_team_accum( list, thread->th.th_serial_team );
            }; // if
        }; // for gtid
    } else {
        __kmp_printf( "Threads array is not allocated.\n" );
    }; // if

    // Print out __kmp_root array.
    __kmp_printf( "\n------------------------------\nUbers\n------------------------------\n" );
    if ( __kmp_root != NULL ) {
        int gtid;
        for ( gtid = 0; gtid < __kmp_threads_capacity; ++ gtid ) {
            kmp_root_t const * root = __kmp_root[ gtid ];
            if ( root != NULL ) {
                __kmp_printf( "GTID %2d %p:\n", gtid, root );
                __kmp_print_structure_team(   "    Root Team:    ",      root->r.r_root_team );
                __kmp_print_structure_team(   "    Hot Team:     ",      root->r.r_hot_team );
                __kmp_print_structure_thread( "    Uber Thread:  ",      root->r.r_uber_thread );
                __kmp_printf(                 "    Active?:      %2d\n", root->r.r_active );
                __kmp_printf(                 "    Nested?:      %2d\n", root->r.r_nested );
                __kmp_printf(                 "    In Parallel:  %2d\n", root->r.r_in_parallel );
                __kmp_printf( "\n" );
                __kmp_print_structure_team_accum( list, root->r.r_root_team );
                __kmp_print_structure_team_accum( list, root->r.r_hot_team );
            }; // if
        }; // for gtid
    } else {
        __kmp_printf( "Ubers array is not allocated.\n" );
    }; // if

    __kmp_printf( "\n------------------------------\nTeams\n------------------------------\n" );
    while ( list->next != NULL ) {
        kmp_team_p const * team = list->entry;
        int i;
        __kmp_printf( "Team %2x %p:\n", team->t.t_id, team );
        __kmp_print_structure_team( "    Parent Team:      ",      team->t.t_parent );
        __kmp_printf(               "    Master TID:       %2d\n", team->t.t_master_tid );
        __kmp_printf(               "    Max threads:      %2d\n", team->t.t_max_nproc );
        __kmp_printf(               "    Levels of serial: %2d\n", team->t.t_serialized );
        __kmp_printf(               "    Number threads:   %2d\n", team->t.t_nproc );
        for ( i = 0; i < team->t.t_nproc; ++ i ) {
            __kmp_printf(           "    Thread %2d:      ", i );
            __kmp_print_structure_thread( "", team->t.t_threads[ i ] );
        }; // for i
        __kmp_print_structure_team( "    Next in pool:     ",      team->t.t_next_pool );
        __kmp_printf( "\n" );
        list = list->next;
    }; // while

    // Print out __kmp_thread_pool and __kmp_team_pool.
    __kmp_printf( "\n------------------------------\nPools\n------------------------------\n" );
    __kmp_print_structure_thread(   "Thread pool:          ", (kmp_info_t *)__kmp_thread_pool );
    __kmp_print_structure_team(     "Team pool:            ", (kmp_team_t *)__kmp_team_pool );
    __kmp_printf( "\n" );

    // Free team list.
    while ( list != NULL ) {
        kmp_team_list_item_t * item = list;
        list = list->next;
        KMP_INTERNAL_FREE( item );
    }; // while

}

#endif


//---------------------------------------------------------------------------
//  Stuff for per-thread fast random number generator
//  Table of primes

static const unsigned __kmp_primes[] = {
  0x9e3779b1, 0xffe6cc59, 0x2109f6dd, 0x43977ab5,
  0xba5703f5, 0xb495a877, 0xe1626741, 0x79695e6b,
  0xbc98c09f, 0xd5bee2b3, 0x287488f9, 0x3af18231,
  0x9677cd4d, 0xbe3a6929, 0xadc6a877, 0xdcf0674b,
  0xbe4d6fe9, 0x5f15e201, 0x99afc3fd, 0xf3f16801,
  0xe222cfff, 0x24ba5fdb, 0x0620452d, 0x79f149e3,
  0xc8b93f49, 0x972702cd, 0xb07dd827, 0x6c97d5ed,
  0x085a3d61, 0x46eb5ea7, 0x3d9910ed, 0x2e687b5b,
  0x29609227, 0x6eb081f1, 0x0954c4e1, 0x9d114db9,
  0x542acfa9, 0xb3e6bd7b, 0x0742d917, 0xe9f3ffa7,
  0x54581edb, 0xf2480f45, 0x0bb9288f, 0xef1affc7,
  0x85fa0ca7, 0x3ccc14db, 0xe6baf34b, 0x343377f7,
  0x5ca19031, 0xe6d9293b, 0xf0a9f391, 0x5d2e980b,
  0xfc411073, 0xc3749363, 0xb892d829, 0x3549366b,
  0x629750ad, 0xb98294e5, 0x892d9483, 0xc235baf3,
  0x3d2402a3, 0x6bdef3c9, 0xbec333cd, 0x40c9520f
};

//---------------------------------------------------------------------------
//  __kmp_get_random: Get a random number using a linear congruential method.

unsigned short
__kmp_get_random( kmp_info_t * thread )
{
  unsigned x = thread -> th.th_x;
  unsigned short r = x>>16;

  thread -> th.th_x = x*thread->th.th_a+1;

  KA_TRACE(30, ("__kmp_get_random: THREAD: %d, RETURN: %u\n",
         thread->th.th_info.ds.ds_tid, r) );

  return r;
}
//--------------------------------------------------------
// __kmp_init_random: Initialize a random number generator

void
__kmp_init_random( kmp_info_t * thread )
{
  unsigned seed = thread->th.th_info.ds.ds_tid;

  thread -> th.th_a = __kmp_primes[seed%(sizeof(__kmp_primes)/sizeof(__kmp_primes[0]))];
  thread -> th.th_x = (seed+1)*thread->th.th_a+1;
  KA_TRACE(30, ("__kmp_init_random: THREAD: %u; A: %u\n", seed, thread -> th.th_a) );
}


#if KMP_OS_WINDOWS
/* reclaim array entries for root threads that are already dead, returns number reclaimed */
static int
__kmp_reclaim_dead_roots(void) {
    int i, r = 0;

    for(i = 0; i < __kmp_threads_capacity; ++i) {
        if( KMP_UBER_GTID( i ) &&
          !__kmp_still_running((kmp_info_t *)TCR_SYNC_PTR(__kmp_threads[i])) &&
          !__kmp_root[i]->r.r_active ) { // AC: reclaim only roots died in non-active state
            r += __kmp_unregister_root_other_thread(i);
        }
    }
    return r;
}
#endif

/*
   This function attempts to create free entries in __kmp_threads and __kmp_root, and returns the number of
   free entries generated.

   For Windows* OS static library, the first mechanism used is to reclaim array entries for root threads that are
   already dead.

   On all platforms, expansion is attempted on the arrays __kmp_threads_ and __kmp_root, with appropriate
   update to __kmp_threads_capacity.  Array capacity is increased by doubling with clipping to
    __kmp_tp_capacity, if threadprivate cache array has been created.
   Synchronization with __kmpc_threadprivate_cached is done using __kmp_tp_cached_lock.

   After any dead root reclamation, if the clipping value allows array expansion to result in the generation
   of a total of nWish free slots, the function does that expansion.  If not, but the clipping value allows
   array expansion to result in the generation of a total of nNeed free slots, the function does that expansion.
   Otherwise, nothing is done beyond the possible initial root thread reclamation.  However, if nNeed is zero,
   a best-effort attempt is made to fulfil nWish as far as possible, i.e. the function will attempt to create
   as many free slots as possible up to nWish.

   If any argument is negative, the behavior is undefined.
*/
static int
__kmp_expand_threads(int nWish, int nNeed) {
    int added = 0;
    int old_tp_cached;
    int __kmp_actual_max_nth;

    if(nNeed > nWish) /* normalize the arguments */
        nWish = nNeed;
#if KMP_OS_WINDOWS && !defined GUIDEDLL_EXPORTS
/* only for Windows static library */
    /* reclaim array entries for root threads that are already dead */
    added = __kmp_reclaim_dead_roots();

    if(nNeed) {
        nNeed -= added;
        if(nNeed < 0)
            nNeed = 0;
    }
    if(nWish) {
        nWish -= added;
        if(nWish < 0)
            nWish = 0;
    }
#endif
    if(nWish <= 0)
        return added;

    while(1) {
        int nTarget;
        int minimumRequiredCapacity;
        int newCapacity;
        kmp_info_t **newThreads;
        kmp_root_t **newRoot;

        //
        // Note that __kmp_threads_capacity is not bounded by __kmp_max_nth.
        // If __kmp_max_nth is set to some value less than __kmp_sys_max_nth
        // by the user via OMP_THREAD_LIMIT, then __kmp_threads_capacity may
        // become > __kmp_max_nth in one of two ways:
        //
        // 1) The initialization thread (gtid = 0) exits.  __kmp_threads[0]
        //    may not be resused by another thread, so we may need to increase
        //    __kmp_threads_capacity to __kmp_max_threads + 1.
        //
        // 2) New foreign root(s) are encountered.  We always register new
        //    foreign roots.  This may cause a smaller # of threads to be
        //    allocated at subsequent parallel regions, but the worker threads
        //    hang around (and eventually go to sleep) and need slots in the
        //    __kmp_threads[] array.
        //
        // Anyway, that is the reason for moving the check to see if
        // __kmp_max_threads was exceeded into __kmp_reseerve_threads()
        // instead of having it performed here. -BB
        //
        old_tp_cached = __kmp_tp_cached;
        __kmp_actual_max_nth = old_tp_cached ? __kmp_tp_capacity : __kmp_sys_max_nth;
        KMP_DEBUG_ASSERT(__kmp_actual_max_nth >= __kmp_threads_capacity);

        /* compute expansion headroom to check if we can expand and whether to aim for nWish or nNeed */
        nTarget = nWish;
        if(__kmp_actual_max_nth - __kmp_threads_capacity < nTarget) {
            /* can't fulfil nWish, so try nNeed */
            if(nNeed) {
                nTarget = nNeed;
                if(__kmp_actual_max_nth - __kmp_threads_capacity < nTarget) {
                    /* possible expansion too small -- give up */
                    break;
                }
            } else {
                /* best-effort */
                nTarget = __kmp_actual_max_nth - __kmp_threads_capacity;
                if(!nTarget) {
                    /* can expand at all -- give up */
                    break;
                }
            }
        }
        minimumRequiredCapacity = __kmp_threads_capacity + nTarget;

        newCapacity = __kmp_threads_capacity;
        do{
            newCapacity =
                newCapacity <= (__kmp_actual_max_nth >> 1) ?
                (newCapacity << 1) :
                __kmp_actual_max_nth;
        } while(newCapacity < minimumRequiredCapacity);
        newThreads = (kmp_info_t**) __kmp_allocate((sizeof(kmp_info_t*) + sizeof(kmp_root_t*)) * newCapacity + CACHE_LINE);
        newRoot = (kmp_root_t**) ((char*)newThreads + sizeof(kmp_info_t*) * newCapacity );
        memcpy(newThreads, __kmp_threads, __kmp_threads_capacity * sizeof(kmp_info_t*));
        memcpy(newRoot, __kmp_root, __kmp_threads_capacity * sizeof(kmp_root_t*));
        memset(newThreads + __kmp_threads_capacity, 0,
               (newCapacity - __kmp_threads_capacity) * sizeof(kmp_info_t*));
        memset(newRoot + __kmp_threads_capacity, 0,
               (newCapacity - __kmp_threads_capacity) * sizeof(kmp_root_t*));

        if(!old_tp_cached && __kmp_tp_cached && newCapacity > __kmp_tp_capacity) {
            /* __kmp_tp_cached has changed, i.e. __kmpc_threadprivate_cached has allocated a threadprivate cache
               while we were allocating the expanded array, and our new capacity is larger than the threadprivate
               cache capacity, so we should deallocate the expanded arrays and try again.  This is the first check
               of a double-check pair.
            */
            __kmp_free(newThreads);
            continue; /* start over and try again */
        }
        __kmp_acquire_bootstrap_lock(&__kmp_tp_cached_lock);
        if(!old_tp_cached && __kmp_tp_cached && newCapacity > __kmp_tp_capacity) {
            /* Same check as above, but this time with the lock so we can be sure if we can succeed. */
            __kmp_release_bootstrap_lock(&__kmp_tp_cached_lock);
            __kmp_free(newThreads);
            continue; /* start over and try again */
        } else {
            /* success */
            // __kmp_free( __kmp_threads ); // ATT: It leads to crash. Need to be investigated.
            //
            *(kmp_info_t**volatile*)&__kmp_threads = newThreads;
            *(kmp_root_t**volatile*)&__kmp_root = newRoot;
            added += newCapacity - __kmp_threads_capacity;
            *(volatile int*)&__kmp_threads_capacity = newCapacity;
            __kmp_release_bootstrap_lock(&__kmp_tp_cached_lock);
            break; /* succeeded, so we can exit the loop */
        }
    }
    return added;
}

/* register the current thread as a root thread and obtain our gtid */
/* we must have the __kmp_initz_lock held at this point */
/* Argument TRUE only if are the thread that calls from __kmp_do_serial_initialize() */
int
__kmp_register_root( int initial_thread )
{
    kmp_info_t *root_thread;
    kmp_root_t *root;
    int         gtid;
    int         capacity;
    __kmp_acquire_bootstrap_lock( &__kmp_forkjoin_lock );
    KA_TRACE( 20, ("__kmp_register_root: entered\n"));
    KMP_MB();


    /*
        2007-03-02:

        If initial thread did not invoke OpenMP RTL yet, and this thread is not an initial one,
        "__kmp_all_nth >= __kmp_threads_capacity" condition does not work as expected -- it may
        return false (that means there is at least one empty slot in __kmp_threads array), but it
        is possible the only free slot is #0, which is reserved for initial thread and so cannot be
        used for this one. Following code workarounds this bug.

        However, right solution seems to be not reserving slot #0 for initial thread because:
            (1) there is no magic in slot #0,
            (2) we cannot detect initial thread reliably (the first thread which does serial
                initialization may be not a real initial thread).
    */
    capacity = __kmp_threads_capacity;
    if ( ! initial_thread && TCR_PTR(__kmp_threads[0]) == NULL ) {
        -- capacity;
    }; // if

    /* see if there are too many threads */
    if ( __kmp_all_nth >= capacity && !__kmp_expand_threads( 1, 1 ) ) {
        if ( __kmp_tp_cached ) {
            __kmp_msg(
                kmp_ms_fatal,
                KMP_MSG( CantRegisterNewThread ),
                KMP_HNT( Set_ALL_THREADPRIVATE, __kmp_tp_capacity ),
                KMP_HNT( PossibleSystemLimitOnThreads ),
                __kmp_msg_null
            );
        }
        else {
            __kmp_msg(
                kmp_ms_fatal,
                KMP_MSG( CantRegisterNewThread ),
                KMP_HNT( SystemLimitOnThreads ),
                __kmp_msg_null
            );
        }
    }; // if

    /* find an available thread slot */
    /* Don't reassign the zero slot since we need that to only be used by initial
       thread */
    for( gtid=(initial_thread ? 0 : 1) ; TCR_PTR(__kmp_threads[gtid]) != NULL ; gtid++ );
    KA_TRACE( 1, ("__kmp_register_root: found slot in threads array: T#%d\n", gtid ));
    KMP_ASSERT( gtid < __kmp_threads_capacity );

    /* update global accounting */
    __kmp_all_nth ++;
    TCW_4(__kmp_nth, __kmp_nth + 1);

    //
    // if __kmp_adjust_gtid_mode is set, then we use method #1 (sp search)
    // for low numbers of procs, and method #2 (keyed API call) for higher
    // numbers of procs.
    //
    if ( __kmp_adjust_gtid_mode ) {
        if ( __kmp_all_nth >= __kmp_tls_gtid_min ) {
            if ( TCR_4(__kmp_gtid_mode) != 2) {
                TCW_4(__kmp_gtid_mode, 2);
            }
        }
        else {
            if (TCR_4(__kmp_gtid_mode) != 1 ) {
                TCW_4(__kmp_gtid_mode, 1);
            }
        }
    }

#ifdef KMP_ADJUST_BLOCKTIME
    /* Adjust blocktime to zero if necessary            */
    /* Middle initialization might not have occurred yet */
    if ( !__kmp_env_blocktime && ( __kmp_avail_proc > 0 ) ) {
        if ( __kmp_nth > __kmp_avail_proc ) {
            __kmp_zero_bt = TRUE;
        }
    }
#endif /* KMP_ADJUST_BLOCKTIME */

    /* setup this new hierarchy */
    if( ! ( root = __kmp_root[gtid] )) {
        root = __kmp_root[gtid] = (kmp_root_t*) __kmp_allocate( sizeof(kmp_root_t) );
        KMP_DEBUG_ASSERT( ! root->r.r_root_team );
    }

    __kmp_initialize_root( root );

    /* setup new root thread structure */
    if( root -> r.r_uber_thread ) {
        root_thread = root -> r.r_uber_thread;
    } else {
        root_thread = (kmp_info_t*) __kmp_allocate( sizeof(kmp_info_t) );
        if ( __kmp_storage_map ) {
            __kmp_print_thread_storage_map( root_thread, gtid );
        }
        root_thread -> th.th_info .ds.ds_gtid = gtid;
        root_thread -> th.th_root =  root;
        if( __kmp_env_consistency_check ) {
            root_thread -> th.th_cons = __kmp_allocate_cons_stack( gtid );
        }
        #if USE_FAST_MEMORY
            __kmp_initialize_fast_memory( root_thread );
        #endif /* USE_FAST_MEMORY */

        #if KMP_USE_BGET
            KMP_DEBUG_ASSERT( root_thread -> th.th_local.bget_data == NULL );
            __kmp_initialize_bget( root_thread );
        #endif
        __kmp_init_random( root_thread );  // Initialize random number generator
    }

    /* setup the serial team held in reserve by the root thread */
    if( ! root_thread -> th.th_serial_team ) {
        #if OMP_30_ENABLED
            kmp_internal_control_t r_icvs = __kmp_get_global_icvs();
        #endif // OMP_30_ENABLED
        KF_TRACE( 10, ( "__kmp_register_root: before serial_team\n" ) );
        root_thread -> th.th_serial_team = __kmp_allocate_team( root, 1, 1,
#if OMP_40_ENABLED
          proc_bind_default,
#endif
#if OMP_30_ENABLED
          &r_icvs,
#else
          __kmp_dflt_team_nth_ub,
          __kmp_global.g.g_dynamic,
          __kmp_dflt_nested,
          __kmp_dflt_blocktime,
          __kmp_bt_intervals,
          __kmp_env_blocktime,
#endif // OMP_30_ENABLED
          0 );
    }
    KMP_ASSERT( root_thread -> th.th_serial_team );
    KF_TRACE( 10, ( "__kmp_register_root: after serial_team = %p\n",
      root_thread -> th.th_serial_team ) );

    /* drop root_thread into place */
    TCW_SYNC_PTR(__kmp_threads[gtid], root_thread);

    root -> r.r_root_team -> t.t_threads[0] = root_thread;
    root -> r.r_hot_team  -> t.t_threads[0] = root_thread;
    root_thread -> th.th_serial_team -> t.t_threads[0] = root_thread;
    root_thread -> th.th_serial_team -> t.t_serialized = 0; // AC: the team created in reserve, not for execution (it is unused for now).
    root -> r.r_uber_thread = root_thread;

    /* initialize the thread, get it ready to go */
    __kmp_initialize_info( root_thread, root->r.r_root_team, 0, gtid );

    /* prepare the master thread for get_gtid() */
    __kmp_gtid_set_specific( gtid );
    #ifdef KMP_TDATA_GTID
        __kmp_gtid = gtid;
    #endif
    __kmp_create_worker( gtid, root_thread, __kmp_stksize );
    KMP_DEBUG_ASSERT( __kmp_gtid_get_specific() == gtid );
    TCW_4(__kmp_init_gtid, TRUE);

    KA_TRACE( 20, ("__kmp_register_root: T#%d init T#%d(%d:%d) arrived: join=%u, plain=%u\n",
                    gtid, __kmp_gtid_from_tid( 0, root->r.r_hot_team ),
                    root -> r.r_hot_team -> t.t_id, 0, KMP_INIT_BARRIER_STATE,
                    KMP_INIT_BARRIER_STATE ) );
    { // Initialize barrier data.
        int b;
        for ( b = 0; b < bs_last_barrier; ++ b ) {
            root_thread->th.th_bar[ b ].bb.b_arrived        = KMP_INIT_BARRIER_STATE;
        }; // for
    }
    KMP_DEBUG_ASSERT( root->r.r_hot_team->t.t_bar[ bs_forkjoin_barrier ].b_arrived == KMP_INIT_BARRIER_STATE );


#if KMP_AFFINITY_SUPPORTED
    if ( TCR_4(__kmp_init_middle) ) {
        __kmp_affinity_set_init_mask( gtid, TRUE );
    }
#endif /* KMP_AFFINITY_SUPPORTED */

    __kmp_root_counter ++;

    KMP_MB();
    __kmp_release_bootstrap_lock( &__kmp_forkjoin_lock );

    return gtid;
}

/* Resets a root thread and clear its root and hot teams.
   Returns the number of __kmp_threads entries directly and indirectly freed.
*/
static int
__kmp_reset_root(int gtid, kmp_root_t *root)
{
    kmp_team_t * root_team = root->r.r_root_team;
    kmp_team_t * hot_team  = root->r.r_hot_team;
    int          n         = hot_team->t.t_nproc;
    int i;

    KMP_DEBUG_ASSERT( ! root->r.r_active );

    root->r.r_root_team = NULL;
    root->r.r_hot_team  = NULL;
        // __kmp_free_team() does not free hot teams, so we have to clear r_hot_team before call
        // to __kmp_free_team().
    __kmp_free_team( root, root_team );
    __kmp_free_team( root, hot_team );

#if OMP_30_ENABLED
    //
    // Before we can reap the thread, we need to make certain that all
    // other threads in the teams that had this root as ancestor have stopped trying to steal tasks.
    //
    if ( __kmp_tasking_mode != tskm_immediate_exec ) {
        __kmp_wait_to_unref_task_teams();
    }
#endif /* OMP_30_ENABLED */

    #if KMP_OS_WINDOWS
        /* Close Handle of root duplicated in __kmp_create_worker (tr #62919) */
        KA_TRACE( 10, ("__kmp_reset_root: free handle, th = %p, handle = %" KMP_UINTPTR_SPEC "\n",
            (LPVOID)&(root->r.r_uber_thread->th),
            root->r.r_uber_thread->th.th_info.ds.ds_thread ) );
        __kmp_free_handle( root->r.r_uber_thread->th.th_info.ds.ds_thread );
    #endif /* KMP_OS_WINDOWS */

    TCW_4(__kmp_nth, __kmp_nth - 1); // __kmp_reap_thread will decrement __kmp_all_nth.
    __kmp_reap_thread( root->r.r_uber_thread, 1 );

        // We canot put root thread to __kmp_thread_pool, so we have to reap it istead of freeing.
    root->r.r_uber_thread = NULL;
    /* mark root as no longer in use */
    root -> r.r_begin = FALSE;

    return n;
}

void
__kmp_unregister_root_current_thread( int gtid )
{
    kmp_root_t *root = __kmp_root[gtid];

    KA_TRACE( 1, ("__kmp_unregister_root_current_thread: enter T#%d\n", gtid ));
    KMP_DEBUG_ASSERT( __kmp_threads && __kmp_threads[gtid] );
    KMP_ASSERT( KMP_UBER_GTID( gtid ));
    KMP_ASSERT( root == __kmp_threads[gtid]->th.th_root );
    KMP_ASSERT( root->r.r_active == FALSE );

    /* this lock should be ok, since unregister_root_current_thread is never called during
     * and abort, only during a normal close.  furthermore, if you have the
     * forkjoin lock, you should never try to get the initz lock */

    __kmp_acquire_bootstrap_lock( &__kmp_forkjoin_lock );

    KMP_MB();

    __kmp_reset_root(gtid, root);

    /* free up this thread slot */
    __kmp_gtid_set_specific( KMP_GTID_DNE );
#ifdef KMP_TDATA_GTID
    __kmp_gtid = KMP_GTID_DNE;
#endif

    KMP_MB();
    KC_TRACE( 10, ("__kmp_unregister_root_current_thread: T#%d unregistered\n", gtid ));

    __kmp_release_bootstrap_lock( &__kmp_forkjoin_lock );
}

/* __kmp_forkjoin_lock must be already held
   Unregisters a root thread that is not the current thread.  Returns the number of
   __kmp_threads entries freed as a result.
 */
static int
__kmp_unregister_root_other_thread( int gtid )
{
    kmp_root_t *root = __kmp_root[gtid];
    int r;

    KA_TRACE( 1, ("__kmp_unregister_root_other_thread: enter T#%d\n", gtid ));
    KMP_DEBUG_ASSERT( __kmp_threads && __kmp_threads[gtid] );
    KMP_ASSERT( KMP_UBER_GTID( gtid ));
    KMP_ASSERT( root == __kmp_threads[gtid]->th.th_root );
    KMP_ASSERT( root->r.r_active == FALSE );

    r = __kmp_reset_root(gtid, root);
    KC_TRACE( 10, ("__kmp_unregister_root_other_thread: T#%d unregistered\n", gtid ));
    return r;
}

#if OMP_30_ENABLED

#if KMP_DEBUG
void __kmp_task_info() {

    kmp_int32 gtid       = __kmp_entry_gtid();
    kmp_int32 tid        = __kmp_tid_from_gtid( gtid );
    kmp_info_t *this_thr = __kmp_threads[ gtid ];
    kmp_team_t *steam    = this_thr -> th.th_serial_team;
    kmp_team_t *team     = this_thr -> th.th_team;

    __kmp_printf( "__kmp_task_info: gtid=%d tid=%d t_thread=%p team=%p curtask=%p ptask=%p\n",
        gtid, tid, this_thr, team, this_thr->th.th_current_task, team->t.t_implicit_task_taskdata[tid].td_parent );
}
#endif // KMP_DEBUG

#endif // OMP_30_ENABLED

/* TODO optimize with one big memclr, take out what isn't needed,
 * split responsility to workers as much as possible, and delay
 * initialization of features as much as possible  */
static void
__kmp_initialize_info( kmp_info_t *this_thr, kmp_team_t *team, int tid, int gtid )
{
    /* this_thr->th.th_info.ds.ds_gtid is setup in kmp_allocate_thread/create_worker
     * this_thr->th.th_serial_team is setup in __kmp_allocate_thread */

    KMP_DEBUG_ASSERT( this_thr != NULL );
    KMP_DEBUG_ASSERT( this_thr -> th.th_serial_team );
    KMP_DEBUG_ASSERT( team );
    KMP_DEBUG_ASSERT( team -> t.t_threads  );
    KMP_DEBUG_ASSERT( team -> t.t_dispatch );
    KMP_DEBUG_ASSERT( team -> t.t_threads[0] );
    KMP_DEBUG_ASSERT( team -> t.t_threads[0] -> th.th_root );

    KMP_MB();

    TCW_SYNC_PTR(this_thr->th.th_team, team);

    this_thr->th.th_info.ds.ds_tid  = tid;
    this_thr->th.th_set_nproc       = 0;
#if OMP_40_ENABLED
    this_thr->th.th_set_proc_bind   = proc_bind_default;
# if KMP_AFFINITY_SUPPORTED
    this_thr->th.th_new_place       = this_thr->th.th_current_place;
# endif
#endif
    this_thr->th.th_root            = team -> t.t_threads[0] -> th.th_root;

    /* setup the thread's cache of the team structure */
    this_thr->th.th_team_nproc      = team -> t.t_nproc;
    this_thr->th.th_team_master     = team -> t.t_threads[0];
    this_thr->th.th_team_serialized = team -> t.t_serialized;
#if OMP_40_ENABLED
    this_thr->th.th_team_microtask  = team -> t.t_threads[0] -> th.th_team_microtask;
    this_thr->th.th_teams_level     = team -> t.t_threads[0] -> th.th_teams_level;
    this_thr->th.th_set_nth_teams   = team -> t.t_threads[0] -> th.th_set_nth_teams;
#endif /* OMP_40_ENABLED */
    TCW_PTR(this_thr->th.th_sleep_loc, NULL);

#if OMP_30_ENABLED
    KMP_DEBUG_ASSERT( team -> t.t_implicit_task_taskdata );
    this_thr->th.th_task_state = 0;

    KF_TRACE( 10, ( "__kmp_initialize_info1: T#%d:%d this_thread=%p curtask=%p\n",
                    tid, gtid, this_thr, this_thr->th.th_current_task ) );

    __kmp_init_implicit_task( this_thr->th.th_team_master->th.th_ident, this_thr, team, tid, TRUE );

    KF_TRACE( 10, ( "__kmp_initialize_info2: T#%d:%d this_thread=%p curtask=%p\n",
                    tid, gtid, this_thr, this_thr->th.th_current_task ) );
    // TODO: Initialize ICVs from parent; GEH - isn't that already done in __kmp_initialize_team()?
#endif // OMP_30_ENABLED

    /* TODO no worksharing in speculative threads */
    this_thr -> th.th_dispatch      = &team -> t.t_dispatch[ tid ];

    this_thr->th.th_local.this_construct = 0;
    this_thr->th.th_local.last_construct = 0;

#ifdef BUILD_TV
    this_thr->th.th_local.tv_data = 0;
#endif

    if ( ! this_thr->th.th_pri_common ) {
        this_thr->th.th_pri_common = (struct common_table *) __kmp_allocate( sizeof(struct common_table) );
        if ( __kmp_storage_map ) {
            __kmp_print_storage_map_gtid(
                gtid, this_thr->th.th_pri_common, this_thr->th.th_pri_common + 1,
                sizeof( struct common_table ), "th_%d.th_pri_common\n", gtid
            );
        }; // if
        this_thr->th.th_pri_head = NULL;
    }; // if

    /* Initialize dynamic dispatch */
    {
        volatile kmp_disp_t *dispatch = this_thr -> th.th_dispatch;
        /*
         * Use team max_nproc since this will never change for the team.
         */
        size_t disp_size = sizeof( dispatch_private_info_t ) *
            ( team->t.t_max_nproc == 1 ? 1 : KMP_MAX_DISP_BUF );
        KD_TRACE( 10, ("__kmp_initialize_info: T#%d max_nproc: %d\n", gtid, team->t.t_max_nproc ) );
        KMP_ASSERT( dispatch );
        KMP_DEBUG_ASSERT( team -> t.t_dispatch );
        KMP_DEBUG_ASSERT( dispatch == &team->t.t_dispatch[ tid ] );

        dispatch->th_disp_index = 0;

        if( ! dispatch -> th_disp_buffer )  {
            dispatch -> th_disp_buffer = (dispatch_private_info_t *) __kmp_allocate( disp_size );

            if ( __kmp_storage_map ) {
                __kmp_print_storage_map_gtid( gtid, &dispatch->th_disp_buffer[ 0 ],
                                         &dispatch->th_disp_buffer[ team->t.t_max_nproc == 1 ? 1 : KMP_MAX_DISP_BUF ],
                                         disp_size, "th_%d.th_dispatch.th_disp_buffer "
                                         "(team_%d.t_dispatch[%d].th_disp_buffer)",
                                         gtid, team->t.t_id, gtid );
            }
        } else {
            memset( & dispatch -> th_disp_buffer[0], '\0', disp_size );
        }

        dispatch -> th_dispatch_pr_current = 0;
        dispatch -> th_dispatch_sh_current = 0;

        dispatch -> th_deo_fcn = 0;             /* ORDERED     */
        dispatch -> th_dxo_fcn = 0;             /* END ORDERED */
    }

    this_thr->th.th_next_pool = NULL;

    KMP_DEBUG_ASSERT( !this_thr->th.th_spin_here );
    KMP_DEBUG_ASSERT( this_thr->th.th_next_waiting == 0 );

    KMP_MB();
}


/* allocate a new thread for the requesting team.  this is only called from within a
 * forkjoin critical section.  we will first try to get an available thread from the
 * thread pool.  if none is available, we will fork a new one assuming we are able
 * to create a new one.  this should be assured, as the caller should check on this
 * first.
 */
kmp_info_t *
__kmp_allocate_thread( kmp_root_t *root, kmp_team_t *team, int new_tid )
{
    kmp_team_t  *serial_team;
    kmp_info_t  *new_thr;
    int          new_gtid;

    KA_TRACE( 20, ("__kmp_allocate_thread: T#%d\n", __kmp_get_gtid() ));
    KMP_DEBUG_ASSERT( root && team );
    KMP_DEBUG_ASSERT( KMP_MASTER_GTID( __kmp_get_gtid() ));
    KMP_MB();

    /* first, try to get one from the thread pool */
    if ( __kmp_thread_pool ) {

        new_thr = (kmp_info_t*)__kmp_thread_pool;
        __kmp_thread_pool = (volatile kmp_info_t *) new_thr->th.th_next_pool;
        if ( new_thr == __kmp_thread_pool_insert_pt ) {
            __kmp_thread_pool_insert_pt = NULL;
        }
        TCW_4(new_thr->th.th_in_pool, FALSE);
        //
        // Don't touch th_active_in_pool or th_active.
        // The worker thread adjusts those flags as it sleeps/awakens.
        //

        __kmp_thread_pool_nth--;

        KA_TRACE( 20, ("__kmp_allocate_thread: T#%d using thread T#%d\n",
                    __kmp_get_gtid(), new_thr->th.th_info.ds.ds_gtid ));
        KMP_ASSERT(       ! new_thr -> th.th_team );
        KMP_DEBUG_ASSERT( __kmp_nth < __kmp_threads_capacity );
        KMP_DEBUG_ASSERT( __kmp_thread_pool_nth >= 0 );

        /* setup the thread structure */
        __kmp_initialize_info( new_thr, team, new_tid, new_thr->th.th_info.ds.ds_gtid );
        KMP_DEBUG_ASSERT( new_thr->th.th_serial_team );

        TCW_4(__kmp_nth, __kmp_nth + 1);

#ifdef KMP_ADJUST_BLOCKTIME
        /* Adjust blocktime back to zero if necessar      y */
        /* Middle initialization might not have occurred yet */
        if ( !__kmp_env_blocktime && ( __kmp_avail_proc > 0 ) ) {
            if ( __kmp_nth > __kmp_avail_proc ) {
                __kmp_zero_bt = TRUE;
            }
        }
#endif /* KMP_ADJUST_BLOCKTIME */

        KF_TRACE( 10, ("__kmp_allocate_thread: T#%d using thread %p T#%d\n",
                    __kmp_get_gtid(), new_thr, new_thr->th.th_info.ds.ds_gtid ));

        KMP_MB();
        return new_thr;
    }


    /* no, well fork a new one */
    KMP_ASSERT( __kmp_nth    == __kmp_all_nth );
    KMP_ASSERT( __kmp_all_nth < __kmp_threads_capacity );

    //
    // If this is the first worker thread the RTL is creating, then also
    // launch the monitor thread.  We try to do this as early as possible.
    //
    if ( ! TCR_4( __kmp_init_monitor ) ) {
        __kmp_acquire_bootstrap_lock( & __kmp_monitor_lock );
        if ( ! TCR_4( __kmp_init_monitor ) ) {
            KF_TRACE( 10, ( "before __kmp_create_monitor\n" ) );
            TCW_4( __kmp_init_monitor, 1 );
            __kmp_create_monitor( & __kmp_monitor );
            KF_TRACE( 10, ( "after __kmp_create_monitor\n" ) );
            #if KMP_OS_WINDOWS
                // AC: wait until monitor has started. This is a fix for CQ232808.
                //     The reason is that if the library is loaded/unloaded in a loop with small (parallel)
                //     work in between, then there is high probability that monitor thread started after
                //     the library shutdown. At shutdown it is too late to cope with the problem, because
                //     when the master is in DllMain (process detach) the monitor has no chances to start
                //     (it is blocked), and master has no means to inform the monitor that the library has gone,
                //     because all the memory which the monitor can access is going to be released/reset.
                while ( TCR_4(__kmp_init_monitor) < 2 ) {
                    KMP_YIELD( TRUE );
                }
                KF_TRACE( 10, ( "after monitor thread has started\n" ) );
            #endif
        }
        __kmp_release_bootstrap_lock( & __kmp_monitor_lock );
    }

    KMP_MB();
    for( new_gtid=1 ; TCR_PTR(__kmp_threads[new_gtid]) != NULL; ++new_gtid ) {
        KMP_DEBUG_ASSERT( new_gtid < __kmp_threads_capacity );
    }

    /* allocate space for it. */
    new_thr = (kmp_info_t*) __kmp_allocate( sizeof(kmp_info_t) );

    TCW_SYNC_PTR(__kmp_threads[new_gtid], new_thr);

    if ( __kmp_storage_map ) {
        __kmp_print_thread_storage_map( new_thr, new_gtid );
    }

    /* add the reserve serialized team, initialized from the team's master thread */
    {
    #if OMP_30_ENABLED
    kmp_internal_control_t r_icvs = __kmp_get_x_global_icvs( team );
    #endif // OMP_30_ENABLED
    KF_TRACE( 10, ( "__kmp_allocate_thread: before th_serial/serial_team\n" ) );
    new_thr -> th.th_serial_team = serial_team =
        (kmp_team_t*) __kmp_allocate_team( root, 1, 1,
#if OMP_40_ENABLED
                                           proc_bind_default,
#endif
#if OMP_30_ENABLED
                                           &r_icvs,
#else
                                           team->t.t_set_nproc[0],
                                           team->t.t_set_dynamic[0],
                                           team->t.t_set_nested[0],
                                           team->t.t_set_blocktime[0],
                                           team->t.t_set_bt_intervals[0],
                                           team->t.t_set_bt_set[0],
#endif // OMP_30_ENABLED
                                           0 );
    }
    KMP_ASSERT ( serial_team );
    serial_team -> t.t_serialized = 0;   // AC: the team created in reserve, not for execution (it is unused for now).
    serial_team -> t.t_threads[0] = new_thr;
    KF_TRACE( 10, ( "__kmp_allocate_thread: after th_serial/serial_team : new_thr=%p\n",
      new_thr ) );

    /* setup the thread structures */
    __kmp_initialize_info( new_thr, team, new_tid, new_gtid );

    #if USE_FAST_MEMORY
        __kmp_initialize_fast_memory( new_thr );
    #endif /* USE_FAST_MEMORY */

    #if KMP_USE_BGET
        KMP_DEBUG_ASSERT( new_thr -> th.th_local.bget_data == NULL );
        __kmp_initialize_bget( new_thr );
    #endif

    __kmp_init_random( new_thr );  // Initialize random number generator

    /* Initialize these only once when thread is grabbed for a team allocation */
    KA_TRACE( 20, ("__kmp_allocate_thread: T#%d init go fork=%u, plain=%u\n",
                    __kmp_get_gtid(), KMP_INIT_BARRIER_STATE, KMP_INIT_BARRIER_STATE ));

    new_thr->th.th_bar[ bs_forkjoin_barrier ].bb.b_go = KMP_INIT_BARRIER_STATE;
    new_thr->th.th_bar[ bs_plain_barrier    ].bb.b_go = KMP_INIT_BARRIER_STATE;
    #if KMP_FAST_REDUCTION_BARRIER
    new_thr->th.th_bar[ bs_reduction_barrier ].bb.b_go = KMP_INIT_BARRIER_STATE;
    #endif // KMP_FAST_REDUCTION_BARRIER

    new_thr->th.th_spin_here = FALSE;
    new_thr->th.th_next_waiting = 0;

#if OMP_40_ENABLED && KMP_AFFINITY_SUPPORTED
    new_thr->th.th_current_place = KMP_PLACE_UNDEFINED;
    new_thr->th.th_new_place = KMP_PLACE_UNDEFINED;
    new_thr->th.th_first_place = KMP_PLACE_UNDEFINED;
    new_thr->th.th_last_place = KMP_PLACE_UNDEFINED;
#endif

    TCW_4(new_thr->th.th_in_pool, FALSE);
    new_thr->th.th_active_in_pool = FALSE;
    TCW_4(new_thr->th.th_active, TRUE);

    /* adjust the global counters */
    __kmp_all_nth ++;
    __kmp_nth ++;

    //
    // if __kmp_adjust_gtid_mode is set, then we use method #1 (sp search)
    // for low numbers of procs, and method #2 (keyed API call) for higher
    // numbers of procs.
    //
    if ( __kmp_adjust_gtid_mode ) {
        if ( __kmp_all_nth >= __kmp_tls_gtid_min ) {
            if ( TCR_4(__kmp_gtid_mode) != 2) {
                TCW_4(__kmp_gtid_mode, 2);
            }
        }
        else {
            if (TCR_4(__kmp_gtid_mode) != 1 ) {
                TCW_4(__kmp_gtid_mode, 1);
            }
        }
    }

#ifdef KMP_ADJUST_BLOCKTIME
    /* Adjust blocktime back to zero if necessary       */
    /* Middle initialization might not have occurred yet */
    if ( !__kmp_env_blocktime && ( __kmp_avail_proc > 0 ) ) {
        if ( __kmp_nth > __kmp_avail_proc ) {
            __kmp_zero_bt = TRUE;
        }
    }
#endif /* KMP_ADJUST_BLOCKTIME */

    /* actually fork it and create the new worker thread */
    KF_TRACE( 10, ("__kmp_allocate_thread: before __kmp_create_worker: %p\n", new_thr ));
    __kmp_create_worker( new_gtid, new_thr, __kmp_stksize );
    KF_TRACE( 10, ("__kmp_allocate_thread: after __kmp_create_worker: %p\n", new_thr ));


    KA_TRACE( 20, ("__kmp_allocate_thread: T#%d forked T#%d\n", __kmp_get_gtid(), new_gtid ));
    KMP_MB();
    return new_thr;
}

/*
 * reinitialize team for reuse.
 *
 * The hot team code calls this case at every fork barrier, so EPCC barrier
 * test are extremely sensitive to changes in it, esp. writes to the team
 * struct, which cause a cache invalidation in all threads.
 *
 * IF YOU TOUCH THIS ROUTINE, RUN EPCC C SYNCBENCH ON A BIG-IRON MACHINE!!!
 */
static void
__kmp_reinitialize_team( kmp_team_t *team,
#if OMP_30_ENABLED
                         kmp_internal_control_t *new_icvs, ident_t *loc
#else
                         int new_set_nproc, int new_set_dynamic, int new_set_nested,
                         int new_set_blocktime, int new_bt_intervals, int new_bt_set
#endif
                         ) {
    KF_TRACE( 10, ( "__kmp_reinitialize_team: enter this_thread=%p team=%p\n",
                    team->t.t_threads[0], team ) );
#if OMP_30_ENABLED
    KMP_DEBUG_ASSERT( team && new_icvs);
    KMP_DEBUG_ASSERT( ( ! TCR_4(__kmp_init_parallel) ) || new_icvs->nproc );
    team->t.t_ident = loc;
#else
    KMP_DEBUG_ASSERT( team && new_set_nproc );
#endif // OMP_30_ENABLED

    team->t.t_id = KMP_GEN_TEAM_ID();

    // Copy ICVs to the master thread's implicit taskdata
#if OMP_30_ENABLED
    load_icvs(new_icvs);
    __kmp_init_implicit_task( loc, team->t.t_threads[0], team, 0, FALSE );
    store_icvs(&team->t.t_implicit_task_taskdata[0].td_icvs, new_icvs);
    sync_icvs();
# else
    team -> t.t_set_nproc[0]   = new_set_nproc;
    team -> t.t_set_dynamic[0] = new_set_dynamic;
    team -> t.t_set_nested[0]  = new_set_nested;
    team -> t.t_set_blocktime[0]   = new_set_blocktime;
    team -> t.t_set_bt_intervals[0] = new_bt_intervals;
    team -> t.t_set_bt_set[0]  = new_bt_set;
# endif // OMP_30_ENABLED

    KF_TRACE( 10, ( "__kmp_reinitialize_team: exit this_thread=%p team=%p\n",
                    team->t.t_threads[0], team ) );
}

static void
__kmp_setup_icv_copy(kmp_team_t *  team, int           new_nproc,
#if OMP_30_ENABLED
                kmp_internal_control_t * new_icvs,
                ident_t *                loc
#else
                int new_set_nproc, int new_set_dynamic, int new_set_nested,
                int new_set_blocktime, int new_bt_intervals, int new_bt_set
#endif // OMP_30_ENABLED
                )
{
    int f;

#if OMP_30_ENABLED
    KMP_DEBUG_ASSERT( team && new_nproc && new_icvs );
    KMP_DEBUG_ASSERT( ( ! TCR_4(__kmp_init_parallel) ) || new_icvs->nproc );
#else
    KMP_DEBUG_ASSERT( team && new_nproc && new_set_nproc );
#endif // OMP_30_ENABLED

    // Master thread's copy of the ICVs was set up on the implicit taskdata in __kmp_reinitialize_team.
    // __kmp_fork_call() assumes the master thread's implicit task has this data before this function is called.
#if KMP_BARRIER_ICV_PULL
    // Copy the ICVs to master's thread structure into th_fixed_icvs (which remains untouched), where all of the
    // worker threads can access them and make their own copies after the barrier.
    load_icvs(new_icvs);
    KMP_DEBUG_ASSERT(team->t.t_threads[0]);  // the threads arrays should be allocated at this point
    store_icvs(&team->t.t_threads[0]->th.th_fixed_icvs, new_icvs);
    sync_icvs();
    KF_TRACE(10, ("__kmp_setup_icv_copy: PULL: T#%d this_thread=%p team=%p\n", 0, team->t.t_threads[0], team));

#elif KMP_BARRIER_ICV_PUSH
    // The ICVs will be propagated in the fork barrier, so nothing needs to be done here.
    KF_TRACE(10, ("__kmp_setup_icv_copy: PUSH: T#%d this_thread=%p team=%p\n", 0, team->t.t_threads[0], team));

#else
    // Copy the ICVs to each of the non-master threads.  This takes O(nthreads) time.
# if OMP_30_ENABLED
    load_icvs(new_icvs);
# endif // OMP_30_ENABLED
    KMP_DEBUG_ASSERT(team->t.t_threads[0]);  // the threads arrays should be allocated at this point
    for(f=1 ; f<new_nproc ; f++) { // skip the master thread
# if OMP_30_ENABLED
        // TODO: GEH - pass in better source location info since usually NULL here
        KF_TRACE( 10, ( "__kmp_setup_icv_copy: LINEAR: T#%d this_thread=%p team=%p\n",
                        f, team->t.t_threads[f], team ) );
        __kmp_init_implicit_task( loc, team->t.t_threads[f], team, f, FALSE );
        store_icvs(&team->t.t_implicit_task_taskdata[f].td_icvs, new_icvs);
        KF_TRACE( 10, ( "__kmp_setup_icv_copy: LINEAR: T#%d this_thread=%p team=%p\n",
                        f, team->t.t_threads[f], team ) );
# else
        team -> t.t_set_nproc[f]   = new_set_nproc;
        team -> t.t_set_dynamic[f] = new_set_dynamic;
        team -> t.t_set_nested[f]  = new_set_nested;
        team -> t.t_set_blocktime[f]   = new_set_blocktime;
        team -> t.t_set_bt_intervals[f] = new_bt_intervals;
        team -> t.t_set_bt_set[f]  = new_bt_set;
# endif // OMP_30_ENABLED
    }
# if OMP_30_ENABLED
    sync_icvs();
# endif // OMP_30_ENABLED
#endif // KMP_BARRIER_ICV_PULL
}

/* initialize the team data structure
 * this assumes the t_threads and t_max_nproc are already set
 * also, we don't touch the arguments */
static void
__kmp_initialize_team(
    kmp_team_t * team,
    int          new_nproc,
    #if OMP_30_ENABLED
        kmp_internal_control_t * new_icvs,
        ident_t *                loc
    #else
        int new_set_nproc, int new_set_dynamic, int new_set_nested,
        int new_set_blocktime, int new_bt_intervals, int new_bt_set
    #endif // OMP_30_ENABLED
) {
    KF_TRACE( 10, ( "__kmp_initialize_team: enter: team=%p\n", team ) );

    /* verify */
    KMP_DEBUG_ASSERT( team );
    KMP_DEBUG_ASSERT( new_nproc <= team->t.t_max_nproc );
    KMP_DEBUG_ASSERT( team->t.t_threads );
    KMP_MB();

    team -> t.t_master_tid  = 0;    /* not needed */
    /* team -> t.t_master_bar;        not needed */
    team -> t.t_serialized  = new_nproc > 1 ? 0 : 1;
    team -> t.t_nproc       = new_nproc;

    /* team -> t.t_parent     = NULL; TODO not needed & would mess up hot team */
    team -> t.t_next_pool   = NULL;
    /* memset( team -> t.t_threads, 0, sizeof(kmp_info_t*)*new_nproc ); would mess up hot team */

    TCW_SYNC_PTR(team->t.t_pkfn, NULL); /* not needed */
    team -> t.t_invoke      = NULL; /* not needed */

#if OMP_30_ENABLED
    // TODO???: team -> t.t_max_active_levels       = new_max_active_levels;
    team -> t.t_sched       = new_icvs->sched;
#endif // OMP_30_ENABLED

#if KMP_ARCH_X86 || KMP_ARCH_X86_64
    team -> t.t_fp_control_saved = FALSE; /* not needed */
    team -> t.t_x87_fpu_control_word = 0; /* not needed */
    team -> t.t_mxcsr = 0;                /* not needed */
#endif /* KMP_ARCH_X86 || KMP_ARCH_X86_64 */

    team -> t.t_construct   = 0;
    __kmp_init_lock( & team -> t.t_single_lock );

    team -> t.t_ordered .dt.t_value = 0;
    team -> t.t_master_active = FALSE;

    memset( & team -> t.t_taskq, '\0', sizeof( kmp_taskq_t ));

#ifdef KMP_DEBUG
    team -> t.t_copypriv_data = NULL;  /* not necessary, but nice for debugging */
#endif
    team -> t.t_copyin_counter = 0;    /* for barrier-free copyin implementation */

    team -> t.t_control_stack_top = NULL;

    __kmp_reinitialize_team( team,
#if OMP_30_ENABLED
                             new_icvs, loc
#else
                             new_set_nproc, new_set_dynamic, new_set_nested,
                             new_set_blocktime, new_bt_intervals, new_bt_set
#endif // OMP_30_ENABLED
                             );


    KMP_MB();
    KF_TRACE( 10, ( "__kmp_initialize_team: exit: team=%p\n", team ) );
}

#if KMP_OS_LINUX && KMP_AFFINITY_SUPPORTED
/* Sets full mask for thread and returns old mask, no changes to structures. */
static void
__kmp_set_thread_affinity_mask_full_tmp( kmp_affin_mask_t *old_mask )
{
    if ( KMP_AFFINITY_CAPABLE() ) {
        int status;
        if ( old_mask != NULL ) {
            status = __kmp_get_system_affinity( old_mask, TRUE );
            int error = errno;
            if ( status != 0 ) {
                __kmp_msg(
                    kmp_ms_fatal,
                    KMP_MSG( ChangeThreadAffMaskError ),
                    KMP_ERR( error ),
                    __kmp_msg_null
                );
            }
        }
        __kmp_set_system_affinity( __kmp_affinity_get_fullMask(), TRUE );
    }
}
#endif

#if OMP_40_ENABLED && KMP_AFFINITY_SUPPORTED

//
// __kmp_partition_places() is the heart of the OpenMP 4.0 affinity mechanism.
// It calculats the worker + master thread's partition based upon the parent
// thread's partition, and binds each worker to a thread in their partition.
// The master thread's partition should already include its current binding.
//
static void
__kmp_partition_places( kmp_team_t *team )
{
    //
    // Copy the master thread's place partion to the team struct
    //
    kmp_info_t *master_th = team->t.t_threads[0];
    KMP_DEBUG_ASSERT( master_th != NULL );
    kmp_proc_bind_t proc_bind = team->t.t_proc_bind;
    int first_place = master_th->th.th_first_place;
    int last_place = master_th->th.th_last_place;
    int masters_place = master_th->th.th_current_place;
    team->t.t_first_place = first_place;
    team->t.t_last_place = last_place;

    KA_TRACE( 20, ("__kmp_partition_places: enter: proc_bind = %d T#%d(%d:0) bound to place %d partition = [%d,%d]\n",
       proc_bind, __kmp_gtid_from_thread( team->t.t_threads[0] ), team->t.t_id,
       masters_place, first_place, last_place ) );

    switch ( proc_bind ) {

        case proc_bind_default:
        //
        // serial teams might have the proc_bind policy set to
        // proc_bind_default.  It doesn't matter, as we don't
        // rebind the master thread for any proc_bind policy.
        //
        KMP_DEBUG_ASSERT( team->t.t_nproc == 1 );
        break;

        case proc_bind_master:
        {
            int f;
            int n_th = team->t.t_nproc;
            for ( f = 1; f < n_th; f++ ) {
                kmp_info_t *th = team->t.t_threads[f];
                KMP_DEBUG_ASSERT( th != NULL );
                th->th.th_first_place = first_place;
                th->th.th_last_place = last_place;
                th->th.th_new_place = masters_place;

                KA_TRACE( 100, ("__kmp_partition_places: master: T#%d(%d:%d) place %d partition = [%d,%d]\n",
                  __kmp_gtid_from_thread( team->t.t_threads[f] ),
                  team->t.t_id, f, masters_place, first_place, last_place ) );
            }
        }
        break;

        case proc_bind_close:
        {
            int f;
            int n_th = team->t.t_nproc;
            int n_places;
            if ( first_place <= last_place ) {
                n_places = last_place - first_place + 1;
            }
            else {
                n_places = __kmp_affinity_num_masks - first_place + last_place + 1;
            }
            if ( n_th <= n_places ) {
                int place = masters_place;
                for ( f = 1; f < n_th; f++ ) {
                    kmp_info_t *th = team->t.t_threads[f];
                    KMP_DEBUG_ASSERT( th != NULL );

                    if ( place == last_place ) {
                        place = first_place;
                    }
                    else if ( place == __kmp_affinity_num_masks - 1) {
                        place = 0;
                    }
                    else {
                        place++;
                    }
                    th->th.th_first_place = first_place;
                    th->th.th_last_place = last_place;
                    th->th.th_new_place = place;

                    KA_TRACE( 100, ("__kmp_partition_places: close: T#%d(%d:%d) place %d partition = [%d,%d]\n",
                       __kmp_gtid_from_thread( team->t.t_threads[f] ),
                       team->t.t_id, f, place, first_place, last_place ) );
                }
            }
            else {
                int S, rem, gap, s_count;
                S = n_th / n_places;
                s_count = 0;
                rem = n_th - ( S * n_places );
                gap = rem > 0 ? n_places/rem : n_places;
                int place = masters_place;
                int gap_ct = gap;
                for ( f = 0; f < n_th; f++ ) {
                    kmp_info_t *th = team->t.t_threads[f];
                    KMP_DEBUG_ASSERT( th != NULL );

                    th->th.th_first_place = first_place;
                    th->th.th_last_place = last_place;
                    th->th.th_new_place = place;
                    s_count++;

                    if ( (s_count == S) && rem && (gap_ct == gap) ) {
                        // do nothing, add an extra thread to place on next iteration
                    }
                    else if ( (s_count == S+1) && rem && (gap_ct == gap) ) {
                        // we added an extra thread to this place; move to next place
                        if ( place == last_place ) {
                            place = first_place;
                        }
                        else if ( place == __kmp_affinity_num_masks - 1) {
                            place = 0;
                        }
                        else {
                            place++;
                        }
                        s_count = 0;
                        gap_ct = 1;
                        rem--;
                    }
                    else if (s_count == S) { // place full; don't add extra
                        if ( place == last_place ) {
                            place = first_place;
                        }
                        else if ( place == __kmp_affinity_num_masks - 1) {
                            place = 0;
                        }
                        else {
                            place++;
                        }
                        gap_ct++;
                        s_count = 0;
                    }

                    KA_TRACE( 100, ("__kmp_partition_places: close: T#%d(%d:%d) place %d partition = [%d,%d]\n",
                      __kmp_gtid_from_thread( team->t.t_threads[f] ),
                      team->t.t_id, f, th->th.th_new_place, first_place,
                      last_place ) );
                }
                KMP_DEBUG_ASSERT( place == masters_place );
            }
        }
        break;

        case proc_bind_spread:
        {
            int f;
            int n_th = team->t.t_nproc;
            int n_places;
            if ( first_place <= last_place ) {
                n_places = last_place - first_place + 1;
            }
            else {
                n_places = __kmp_affinity_num_masks - first_place + last_place + 1;
            }
            if ( n_th <= n_places ) {
                int place = masters_place;
                int S = n_places/n_th;
                int s_count, rem, gap, gap_ct;
                rem = n_places - n_th*S;
                gap = rem ? n_th/rem : 1;
                gap_ct = gap;
                for ( f = 0; f < n_th; f++ ) {
                    kmp_info_t *th = team->t.t_threads[f];
                    KMP_DEBUG_ASSERT( th != NULL );

                    th->th.th_first_place = place;
                    th->th.th_new_place = place;
                    s_count = 1;
                    while (s_count < S) {
                        if ( place == last_place ) {
                            place = first_place;
                        }
                        else if ( place == __kmp_affinity_num_masks - 1) {
                            place = 0;
                        }
                        else {
                            place++;
                        }
                        s_count++;
                    }
                    if (rem && (gap_ct == gap)) {
                        if ( place == last_place ) {
                            place = first_place;
                        }
                        else if ( place == __kmp_affinity_num_masks - 1) {
                            place = 0;
                        }
                        else {
                            place++;
                        }
                        rem--;
                        gap_ct = 0;
                    }
                    th->th.th_last_place = place;
                    gap_ct++;

                    if ( place == last_place ) {
                        place = first_place;
                    }
                    else if ( place == __kmp_affinity_num_masks - 1) {
                        place = 0;
                    }
                    else {
                        place++;
                    }

                    KA_TRACE( 100, ("__kmp_partition_places: spread: T#%d(%d:%d) place %d partition = [%d,%d]\n",
                      __kmp_gtid_from_thread( team->t.t_threads[f] ),
                      team->t.t_id, f, th->th.th_new_place,
                      th->th.th_first_place, th->th.th_last_place ) );
                }
                KMP_DEBUG_ASSERT( place == masters_place );
            }
            else {
                int S, rem, gap, s_count;
                S = n_th / n_places;
                s_count = 0;
                rem = n_th - ( S * n_places );
                gap = rem > 0 ? n_places/rem : n_places;
                int place = masters_place;
                int gap_ct = gap;
                for ( f = 0; f < n_th; f++ ) {
                    kmp_info_t *th = team->t.t_threads[f];
                    KMP_DEBUG_ASSERT( th != NULL );

                    th->th.th_first_place = place;
                    th->th.th_last_place = place;
                    th->th.th_new_place = place;
                    s_count++;

                    if ( (s_count == S) && rem && (gap_ct == gap) ) {
                        // do nothing, add an extra thread to place on next iteration
                    }
                    else if ( (s_count == S+1) && rem && (gap_ct == gap) ) {
                        // we added an extra thread to this place; move on to next place
                        if ( place == last_place ) {
                            place = first_place;
                        }
                        else if ( place == __kmp_affinity_num_masks - 1) {
                            place = 0;
                        }
                        else {
                            place++;
                        }
                        s_count = 0;
                        gap_ct = 1;
                        rem--;
                    }
                    else if (s_count == S) { // place is full; don't add extra thread
                        if ( place == last_place ) {
                            place = first_place;
                        }
                        else if ( place == __kmp_affinity_num_masks - 1) {
                            place = 0;
                        }
                        else {
                            place++;
                        }
                        gap_ct++;
                        s_count = 0;
                    }

                    KA_TRACE( 100, ("__kmp_partition_places: spread: T#%d(%d:%d) place %d partition = [%d,%d]\n",
                       __kmp_gtid_from_thread( team->t.t_threads[f] ),
                       team->t.t_id, f, th->th.th_new_place,
                       th->th.th_first_place, th->th.th_last_place) );
                }
                KMP_DEBUG_ASSERT( place == masters_place );
            }
        }
        break;

        default:
        break;
    }

    KA_TRACE( 20, ("__kmp_partition_places: exit T#%d\n", team->t.t_id ) );
}

#endif /* OMP_40_ENABLED && KMP_AFFINITY_SUPPORTED */

/* allocate a new team data structure to use.  take one off of the free pool if available */
kmp_team_t *
__kmp_allocate_team( kmp_root_t *root, int new_nproc, int max_nproc,
#if OMP_40_ENABLED
    kmp_proc_bind_t new_proc_bind,
#endif
#if OMP_30_ENABLED
    kmp_internal_control_t *new_icvs,
#else
    int new_set_nproc, int new_set_dynamic, int new_set_nested,
    int new_set_blocktime, int new_bt_intervals, int new_bt_set,
#endif
    int argc )
{
    int f;
    kmp_team_t *team;
    char *ptr;
    size_t size;

    KA_TRACE( 20, ("__kmp_allocate_team: called\n"));
    KMP_DEBUG_ASSERT( new_nproc >=1 && argc >=0 );
    KMP_DEBUG_ASSERT( max_nproc >= new_nproc );
    KMP_MB();

    //
    // optimization to use a "hot" team for the top level,
    // as it is usually the same
    //
    if ( ! root->r.r_active  &&  new_nproc > 1 ) {

        KMP_DEBUG_ASSERT( new_nproc == max_nproc );

        team =  root -> r.r_hot_team;

#if OMP_30_ENABLED && KMP_DEBUG
        if ( __kmp_tasking_mode != tskm_immediate_exec ) {
            KA_TRACE( 20, ("__kmp_allocate_team: hot team task_team = %p before reinit\n",
                           team -> t.t_task_team ));
        }
#endif

        /* has the number of threads changed? */
        if( team -> t.t_nproc > new_nproc ) {
            KA_TRACE( 20, ("__kmp_allocate_team: decreasing hot team thread count to %d\n", new_nproc ));

#if KMP_MIC
            team -> t.t_size_changed = 1;
#endif
#if OMP_30_ENABLED
            if ( __kmp_tasking_mode != tskm_immediate_exec ) {
                kmp_task_team_t *task_team = team->t.t_task_team;
                if ( ( task_team != NULL ) && TCR_SYNC_4(task_team->tt.tt_active) ) {
                    //
                    // Signal the worker threads (esp. the extra ones) to stop
                    // looking for tasks while spin waiting.  The task teams
                    // are reference counted and will be deallocated by the
                    // last worker thread.
                    //
                    KMP_DEBUG_ASSERT( team->t.t_nproc > 1 );
                    TCW_SYNC_4( task_team->tt.tt_active, FALSE );
                    KMP_MB();

                    KA_TRACE( 20, ( "__kmp_allocate_team: setting task_team %p to NULL\n",
                      &team->t.t_task_team ) );
                      team->t.t_task_team = NULL;
                }
                else {
                    KMP_DEBUG_ASSERT( task_team == NULL );
                }
            }
#endif // OMP_30_ENABLED

            /* release the extra threads we don't need any more */
            for( f = new_nproc  ;  f < team->t.t_nproc  ;  f++ ) {
                KMP_DEBUG_ASSERT( team->t.t_threads[ f ] );
                __kmp_free_thread( team->t.t_threads[ f ] );
                team -> t.t_threads[ f ] =  NULL;
            }

            team -> t.t_nproc =  new_nproc;
#if OMP_30_ENABLED
            // TODO???: team -> t.t_max_active_levels = new_max_active_levels;
            team -> t.t_sched =  new_icvs->sched;
#endif
            __kmp_reinitialize_team( team,
#if OMP_30_ENABLED
                                     new_icvs, root->r.r_uber_thread->th.th_ident
#else
                                     new_set_nproc, new_set_dynamic, new_set_nested,
                                     new_set_blocktime, new_bt_intervals, new_bt_set
#endif // OMP_30_ENABLED
                                     );


#if OMP_30_ENABLED
            if ( __kmp_tasking_mode != tskm_immediate_exec ) {
                kmp_task_team_t *task_team = team->t.t_task_team;
                if ( task_team != NULL ) {
                    KMP_DEBUG_ASSERT( ! TCR_4(task_team->tt.tt_found_tasks) );
                    task_team->tt.tt_nproc = new_nproc;
                    task_team->tt.tt_unfinished_threads = new_nproc;
                    task_team->tt.tt_ref_ct = new_nproc - 1;
                }
            }
#endif

            /* update the remaining threads */
            for( f = 0  ;  f < new_nproc  ;  f++ ) {
                team -> t.t_threads[ f ] -> th.th_team_nproc = team->t.t_nproc;
            }

#if OMP_30_ENABLED
            // restore the current task state of the master thread: should be the implicit task
            KF_TRACE( 10, ("__kmp_allocate_team: T#%d, this_thread=%p team=%p\n",
                       0, team->t.t_threads[0], team ) );

            __kmp_push_current_task_to_thread( team -> t.t_threads[ 0 ], team, 0 );
#endif

#ifdef KMP_DEBUG
            for ( f = 0; f < team->t.t_nproc; f++ ) {
                KMP_DEBUG_ASSERT( team->t.t_threads[f] &&
                    team->t.t_threads[f]->th.th_team_nproc == team->t.t_nproc );
            }
#endif

#if OMP_40_ENABLED
            team->t.t_proc_bind = new_proc_bind;
# if KMP_AFFINITY_SUPPORTED
            __kmp_partition_places( team );
# endif
#endif

        }
        else if ( team -> t.t_nproc < new_nproc ) {
#if KMP_OS_LINUX && KMP_AFFINITY_SUPPORTED
            kmp_affin_mask_t *old_mask;
            if ( KMP_AFFINITY_CAPABLE() ) {
                KMP_CPU_ALLOC(old_mask);
            }
#endif

            KA_TRACE( 20, ("__kmp_allocate_team: increasing hot team thread count to %d\n", new_nproc ));

#if KMP_MIC
            team -> t.t_size_changed = 1;
#endif


            if(team -> t.t_max_nproc < new_nproc) {
                /* reallocate larger arrays */
                __kmp_reallocate_team_arrays(team, new_nproc);
                __kmp_reinitialize_team( team,
#if OMP_30_ENABLED
                                         new_icvs, NULL
#else
                                         new_set_nproc, new_set_dynamic, new_set_nested,
                                         new_set_blocktime, new_bt_intervals, new_bt_set
#endif // OMP_30_ENABLED
                                         );
            }

#if KMP_OS_LINUX && KMP_AFFINITY_SUPPORTED
            /* Temporarily set full mask for master thread before
               creation of workers. The reason is that workers inherit
               the affinity from master, so if a lot of workers are
               created on the single core quickly, they don't get
               a chance to set their own affinity for a long time.
            */
            __kmp_set_thread_affinity_mask_full_tmp( old_mask );
#endif

            /* allocate new threads for the hot team */
            for( f = team->t.t_nproc  ;  f < new_nproc  ;  f++ ) {
                kmp_info_t * new_worker = __kmp_allocate_thread( root, team, f );
                KMP_DEBUG_ASSERT( new_worker );
                team->t.t_threads[ f ] = new_worker;
                new_worker->th.th_team_nproc = team->t.t_nproc;

                KA_TRACE( 20, ("__kmp_allocate_team: team %d init T#%d arrived: join=%u, plain=%u\n",
                                team->t.t_id, __kmp_gtid_from_tid( f, team ), team->t.t_id, f,
                                team->t.t_bar[bs_forkjoin_barrier].b_arrived,
                                team->t.t_bar[bs_plain_barrier].b_arrived ) );

                { // Initialize barrier data for new threads.
                    int b;
                    kmp_balign_t * balign = new_worker->th.th_bar;
                    for ( b = 0; b < bp_last_bar; ++ b ) {
                        balign[ b ].bb.b_arrived        = team->t.t_bar[ b ].b_arrived;
                    }
                }
            }

#if KMP_OS_LINUX && KMP_AFFINITY_SUPPORTED
            if ( KMP_AFFINITY_CAPABLE() ) {
                /* Restore initial master thread's affinity mask */
                __kmp_set_system_affinity( old_mask, TRUE );
                KMP_CPU_FREE(old_mask);
            }
#endif

            /* make sure everyone is syncronized */
            __kmp_initialize_team( team, new_nproc,
#if OMP_30_ENABLED
              new_icvs,
              root->r.r_uber_thread->th.th_ident
#else
              new_set_nproc, new_set_dynamic, new_set_nested,
              new_set_blocktime, new_bt_intervals, new_bt_set
#endif
            );

#if OMP_30_ENABLED
            if ( __kmp_tasking_mode != tskm_immediate_exec ) {
                kmp_task_team_t *task_team = team->t.t_task_team;
                if ( task_team != NULL ) {
                    KMP_DEBUG_ASSERT( ! TCR_4(task_team->tt.tt_found_tasks) );
                    task_team->tt.tt_nproc = new_nproc;
                    task_team->tt.tt_unfinished_threads = new_nproc;
                    task_team->tt.tt_ref_ct = new_nproc - 1;
                }
            }
#endif

            /* reinitialize the old threads */
            for( f = 0  ;  f < team->t.t_nproc  ;  f++ )
                __kmp_initialize_info( team->t.t_threads[ f ], team, f,
                                       __kmp_gtid_from_tid( f, team ) );
#ifdef KMP_DEBUG
            for ( f = 0; f < team->t.t_nproc; ++ f ) {
                KMP_DEBUG_ASSERT( team->t.t_threads[f] &&
                    team->t.t_threads[f]->th.th_team_nproc == team->t.t_nproc );
            }
#endif

#if OMP_40_ENABLED
            team->t.t_proc_bind = new_proc_bind;
# if KMP_AFFINITY_SUPPORTED
            __kmp_partition_places( team );
# endif
#endif

        }
        else {
            KA_TRACE( 20, ("__kmp_allocate_team: reusing hot team\n" ));
#if KMP_MIC
            // This case can mean that omp_set_num_threads() was called and the hot team size
            // was already reduced, so we check the special flag
            if ( team -> t.t_size_changed == -1 ) {
                team -> t.t_size_changed = 1;
            } else {
                team -> t.t_size_changed = 0;
            }
#endif

#if OMP_30_ENABLED
            // TODO???: team -> t.t_max_active_levels = new_max_active_levels;
            team -> t.t_sched =  new_icvs->sched;
#endif

            __kmp_reinitialize_team( team,
#if OMP_30_ENABLED
                                     new_icvs, root->r.r_uber_thread->th.th_ident
#else
                                     new_set_nproc, new_set_dynamic, new_set_nested,
                                     new_set_blocktime, new_bt_intervals, new_bt_set
#endif // OMP_30_ENABLED
                                     );

#if OMP_30_ENABLED
            KF_TRACE( 10, ("__kmp_allocate_team2: T#%d, this_thread=%p team=%p\n",
                           0, team->t.t_threads[0], team ) );
            __kmp_push_current_task_to_thread( team -> t.t_threads[ 0 ], team, 0 );
#endif

#if OMP_40_ENABLED
# if KMP_AFFINITY_SUPPORTED
            if ( team->t.t_proc_bind == new_proc_bind ) {
                KA_TRACE( 200, ("__kmp_allocate_team: reusing hot team #%d bindings: proc_bind = %d, partition = [%d,%d]\n",
                  team->t.t_id, new_proc_bind, team->t.t_first_place,
                  team->t.t_last_place ) );
            }
            else {
                team->t.t_proc_bind = new_proc_bind;
                __kmp_partition_places( team );
            }
# else
            if ( team->t.t_proc_bind != new_proc_bind ) {
                team->t.t_proc_bind = new_proc_bind;
            }
# endif /* KMP_AFFINITY_SUPPORTED */
#endif /* OMP_40_ENABLED */
        }

        /* reallocate space for arguments if necessary */
        __kmp_alloc_argv_entries( argc, team, TRUE );
        team -> t.t_argc     = argc;
        //
        // The hot team re-uses the previous task team,
        // if untouched during the previous release->gather phase.
        //

        KF_TRACE( 10, ( " hot_team = %p\n", team ) );

#if OMP_30_ENABLED && KMP_DEBUG
        if ( __kmp_tasking_mode != tskm_immediate_exec ) {
            KA_TRACE( 20, ("__kmp_allocate_team: hot team task_team = %p after reinit\n",
              team -> t.t_task_team ));
        }
#endif

        KMP_MB();

        return team;
    }

    /* next, let's try to take one from the team pool */
    KMP_MB();
    for( team = (kmp_team_t*) __kmp_team_pool ; (team) ; )
    {
        /* TODO: consider resizing undersized teams instead of reaping them, now that we have a resizing mechanism */
        if ( team->t.t_max_nproc >= max_nproc ) {
            /* take this team from the team pool */
            __kmp_team_pool = team->t.t_next_pool;

            /* setup the team for fresh use */
            __kmp_initialize_team( team, new_nproc,
#if OMP_30_ENABLED
              new_icvs,
              NULL // TODO: !!!
#else
              new_set_nproc, new_set_dynamic, new_set_nested,
              new_set_blocktime, new_bt_intervals, new_bt_set
#endif
            );

#if OMP_30_ENABLED
            KA_TRACE( 20, ( "__kmp_allocate_team: setting task_team %p to NULL\n",
                            &team->t.t_task_team ) );
            team -> t.t_task_team = NULL;
#endif

            /* reallocate space for arguments if necessary */
            __kmp_alloc_argv_entries( argc, team, TRUE );
            team -> t.t_argc     = argc;

            KA_TRACE( 20, ("__kmp_allocate_team: team %d init arrived: join=%u, plain=%u\n",
                            team->t.t_id, KMP_INIT_BARRIER_STATE, KMP_INIT_BARRIER_STATE ));
            { // Initialize barrier data.
                int b;
                for ( b = 0; b < bs_last_barrier; ++ b) {
                    team->t.t_bar[ b ].b_arrived        = KMP_INIT_BARRIER_STATE;
                }
            }

#if OMP_40_ENABLED
            team->t.t_proc_bind = new_proc_bind;
#endif

            KA_TRACE( 20, ("__kmp_allocate_team: using team from pool %d.\n", team->t.t_id ));
            KMP_MB();

            return team;
        }

        /* reap team if it is too small, then loop back and check the next one */
        /* not sure if this is wise, but, will be redone during the hot-teams rewrite. */
        /* TODO: Use technique to find the right size hot-team, don't reap them */
        team =  __kmp_reap_team( team );
        __kmp_team_pool = team;
    }

    /* nothing available in the pool, no matter, make a new team! */
    KMP_MB();
    team = (kmp_team_t*) __kmp_allocate( sizeof( kmp_team_t ) );

    /* and set it up */
    team -> t.t_max_nproc   = max_nproc;
    /* NOTE well, for some reason allocating one big buffer and dividing it
     * up seems to really hurt performance a lot on the P4, so, let's not use
     * this... */
    __kmp_allocate_team_arrays( team, max_nproc );

    KA_TRACE( 20, ( "__kmp_allocate_team: making a new team\n" ) );
    __kmp_initialize_team( team, new_nproc,
#if OMP_30_ENABLED
      new_icvs,
      NULL // TODO: !!!
#else
      new_set_nproc, new_set_dynamic, new_set_nested,
      new_set_blocktime, new_bt_intervals, new_bt_set
#endif
    );

#if OMP_30_ENABLED
    KA_TRACE( 20, ( "__kmp_allocate_team: setting task_team %p to NULL\n",
                    &team->t.t_task_team ) );
    team -> t.t_task_team = NULL;    // to be removed, as __kmp_allocate zeroes memory, no need to duplicate
#endif

    if ( __kmp_storage_map ) {
        __kmp_print_team_storage_map( "team", team, team->t.t_id, new_nproc );
    }

    /* allocate space for arguments */
    __kmp_alloc_argv_entries( argc, team, FALSE );
    team -> t.t_argc        = argc;

    KA_TRACE( 20, ("__kmp_allocate_team: team %d init arrived: join=%u, plain=%u\n",
                    team->t.t_id, KMP_INIT_BARRIER_STATE, KMP_INIT_BARRIER_STATE ));
    { // Initialize barrier data.
        int b;
        for ( b = 0; b < bs_last_barrier; ++ b ) {
            team->t.t_bar[ b ].b_arrived        = KMP_INIT_BARRIER_STATE;
        }
    }

#if OMP_40_ENABLED
    team->t.t_proc_bind = new_proc_bind;
#endif

    KMP_MB();

    KA_TRACE( 20, ("__kmp_allocate_team: done creating a new team %d.\n", team->t.t_id ));

    return team;
}

/* TODO implement hot-teams at all levels */
/* TODO implement lazy thread release on demand (disband request) */

/* free the team.  return it to the team pool.  release all the threads
 * associated with it */
void
__kmp_free_team( kmp_root_t *root, kmp_team_t *team )
{
    int f;
    KA_TRACE( 20, ("__kmp_free_team: T#%d freeing team %d\n", __kmp_get_gtid(), team->t.t_id ));

    /* verify state */
    KMP_DEBUG_ASSERT( root );
    KMP_DEBUG_ASSERT( team );
    KMP_DEBUG_ASSERT( team->t.t_nproc <= team->t.t_max_nproc );
    KMP_DEBUG_ASSERT( team->t.t_threads );

    /* team is done working */
    TCW_SYNC_PTR(team->t.t_pkfn, NULL); // Important for Debugging Support Library.
    team -> t.t_copyin_counter = 0; // init counter for possible reuse
    // Do not reset pointer to parent team to NULL for hot teams.

    /* if we are a nested team, release our threads */
    if( team != root->r.r_hot_team ) {

#if OMP_30_ENABLED
        if ( __kmp_tasking_mode != tskm_immediate_exec ) {
            kmp_task_team_t *task_team = team->t.t_task_team;
            if ( task_team != NULL ) {
                //
                // Signal the worker threads to stop looking for tasks while
                // spin waiting.  The task teams are reference counted and will
                // be deallocated by the last worker thread via the thread's
                // pointer to the task team.
                //
                KA_TRACE( 20, ( "__kmp_free_team: deactivating task_team %p\n",
                                task_team ) );
                KMP_DEBUG_ASSERT( team->t.t_nproc > 1 );
                TCW_SYNC_4( task_team->tt.tt_active, FALSE );
                KMP_MB();
                team->t.t_task_team = NULL;
            }
        }
#endif /* OMP_30_ENABLED */

        // Reset pointer to parent team only for non-hot teams.
        team -> t.t_parent = NULL;


        /* free the worker threads */
        for ( f = 1; f < team->t.t_nproc; ++ f ) {
            KMP_DEBUG_ASSERT( team->t.t_threads[ f ] );
            __kmp_free_thread( team->t.t_threads[ f ] );
            team->t.t_threads[ f ] = NULL;
        }


        /* put the team back in the team pool */
        /* TODO limit size of team pool, call reap_team if pool too large */
        team -> t.t_next_pool  = (kmp_team_t*) __kmp_team_pool;
        __kmp_team_pool        = (volatile kmp_team_t*) team;
    }

    KMP_MB();
}


/* reap the team.  destroy it, reclaim all its resources and free its memory */
kmp_team_t *
__kmp_reap_team( kmp_team_t *team )
{
    kmp_team_t *next_pool = team -> t.t_next_pool;

    KMP_DEBUG_ASSERT( team );
    KMP_DEBUG_ASSERT( team -> t.t_dispatch    );
    KMP_DEBUG_ASSERT( team -> t.t_disp_buffer );
    KMP_DEBUG_ASSERT( team -> t.t_threads     );
    #if OMP_30_ENABLED
    #else
    KMP_DEBUG_ASSERT( team -> t.t_set_nproc   );
    #endif
    KMP_DEBUG_ASSERT( team -> t.t_argv        );

    /* TODO clean the threads that are a part of this? */

    /* free stuff */

    __kmp_free_team_arrays( team );
#if (KMP_PERF_V106 == KMP_ON)
    if ( team -> t.t_argv != &team -> t.t_inline_argv[0] )
        __kmp_free( (void*) team -> t.t_argv );
#else
    __kmp_free( (void*) team -> t.t_argv );
#endif
    __kmp_free( team );

    KMP_MB();
    return next_pool;
}

//
// Free the thread.  Don't reap it, just place it on the pool of available
// threads.
//
// Changes for Quad issue 527845: We need a predictable OMP tid <-> gtid
// binding for the affinity mechanism to be useful.
//
// Now, we always keep the free list (__kmp_thread_pool) sorted by gtid.
// However, we want to avoid a potential performance problem by always
// scanning through the list to find the correct point at which to insert
// the thread (potential N**2 behavior).  To do this we keep track of the
// last place a thread struct was inserted (__kmp_thread_pool_insert_pt).
// With single-level parallelism, threads will always be added to the tail
// of the list, kept track of by __kmp_thread_pool_insert_pt.  With nested
// parallelism, all bets are off and we may need to scan through the entire
// free list.
//
// This change also has a potentially large performance benefit, for some
// applications.  Previously, as threads were freed from the hot team, they
// would be placed back on the free list in inverse order.  If the hot team
// grew back to it's original size, then the freed thread would be placed
// back on the hot team in reverse order.  This could cause bad cache
// locality problems on programs where the size of the hot team regularly
// grew and shrunk.
//
// Now, for single-level parallelism, the OMP tid is alway == gtid.
//
void
__kmp_free_thread( kmp_info_t *this_th )
{
    int gtid;
    kmp_info_t **scan;

    KA_TRACE( 20, ("__kmp_free_thread: T#%d putting T#%d back on free pool.\n",
                __kmp_get_gtid(), this_th->th.th_info.ds.ds_gtid ));

    KMP_DEBUG_ASSERT( this_th );


    /* put thread back on the free pool */
    TCW_PTR(this_th->th.th_team, NULL);
    TCW_PTR(this_th->th.th_root, NULL);
    TCW_PTR(this_th->th.th_dispatch, NULL);               /* NOT NEEDED */

    //
    // If the __kmp_thread_pool_insert_pt is already past the new insert
    // point, then we need to re-scan the entire list.
    //
    gtid = this_th->th.th_info.ds.ds_gtid;
    if ( __kmp_thread_pool_insert_pt != NULL ) {
        KMP_DEBUG_ASSERT( __kmp_thread_pool != NULL );
        if ( __kmp_thread_pool_insert_pt->th.th_info.ds.ds_gtid > gtid ) {
             __kmp_thread_pool_insert_pt = NULL;
        }
    }

    //
    // Scan down the list to find the place to insert the thread.
    // scan is the address of a link in the list, possibly the address of
    // __kmp_thread_pool itself.
    //
    // In the absence of nested parallism, the for loop will have 0 iterations.
    //
    if ( __kmp_thread_pool_insert_pt != NULL ) {
        scan = &( __kmp_thread_pool_insert_pt->th.th_next_pool );
    }
    else {
        scan = (kmp_info_t **)&__kmp_thread_pool;
    }
    for (; ( *scan != NULL ) && ( (*scan)->th.th_info.ds.ds_gtid < gtid );
      scan = &( (*scan)->th.th_next_pool ) );

    //
    // Insert the new element on the list, and set __kmp_thread_pool_insert_pt
    // to its address.
    //
    TCW_PTR(this_th->th.th_next_pool, *scan);
    __kmp_thread_pool_insert_pt = *scan = this_th;
    KMP_DEBUG_ASSERT( ( this_th->th.th_next_pool == NULL )
      || ( this_th->th.th_info.ds.ds_gtid
      < this_th->th.th_next_pool->th.th_info.ds.ds_gtid ) );
    TCW_4(this_th->th.th_in_pool, TRUE);
    __kmp_thread_pool_nth++;

    TCW_4(__kmp_nth, __kmp_nth - 1);

#ifdef KMP_ADJUST_BLOCKTIME
    /* Adjust blocktime back to user setting or default if necessary */
    /* Middle initialization might never have occurred                */
    if ( !__kmp_env_blocktime && ( __kmp_avail_proc > 0 ) ) {
        KMP_DEBUG_ASSERT( __kmp_avail_proc > 0 );
        if ( __kmp_nth <= __kmp_avail_proc ) {
            __kmp_zero_bt = FALSE;
        }
    }
#endif /* KMP_ADJUST_BLOCKTIME */

    KMP_MB();
}

void
__kmp_join_barrier( int gtid )
{
    register kmp_info_t   *this_thr       = __kmp_threads[ gtid ];
    register kmp_team_t   *team;
    register kmp_uint      nproc;
    kmp_info_t            *master_thread;
    int                    tid;
    #ifdef KMP_DEBUG
        int                    team_id;
    #endif /* KMP_DEBUG */
#if USE_ITT_BUILD
    void * itt_sync_obj = NULL;
    #if USE_ITT_NOTIFY
        if ( __itt_sync_create_ptr || KMP_ITT_DEBUG ) // don't call routine without need
            itt_sync_obj = __kmp_itt_barrier_object( gtid, bs_forkjoin_barrier ); // get object created at fork_barrier
    #endif
#endif /* USE_ITT_BUILD */

    KMP_MB();

    /* get current info */
    team          = this_thr -> th.th_team;
    /*    nproc         = team -> t.t_nproc;*/
    nproc         = this_thr -> th.th_team_nproc;
    KMP_DEBUG_ASSERT( nproc == team->t.t_nproc );
    tid           = __kmp_tid_from_gtid(gtid);
    #ifdef KMP_DEBUG
        team_id       = team -> t.t_id;
    #endif /* KMP_DEBUG */
    /*    master_thread = team -> t.t_threads[0];*/
    master_thread = this_thr -> th.th_team_master;
    #ifdef KMP_DEBUG
        if ( master_thread != team->t.t_threads[0] ) {
            __kmp_print_structure();
        }
    #endif /* KMP_DEBUG */
    KMP_DEBUG_ASSERT( master_thread == team->t.t_threads[0] );
    KMP_MB();

    /* verify state */
    KMP_DEBUG_ASSERT( __kmp_threads && __kmp_threads[gtid] );
    KMP_DEBUG_ASSERT( TCR_PTR(this_thr->th.th_team) );
    KMP_DEBUG_ASSERT( TCR_PTR(this_thr->th.th_root) );
    KMP_DEBUG_ASSERT( this_thr == team -> t.t_threads[tid] );

    KA_TRACE( 10, ("__kmp_join_barrier: T#%d(%d:%d) arrived at join barrier\n",
                   gtid, team_id, tid ));

    #if OMP_30_ENABLED
        if ( __kmp_tasking_mode == tskm_extra_barrier ) {
            __kmp_tasking_barrier( team, this_thr, gtid );

            KA_TRACE( 10, ("__kmp_join_barrier: T#%d(%d:%d) past taking barrier\n",
                           gtid, team_id, tid ));
        }
        #ifdef KMP_DEBUG
        if ( __kmp_tasking_mode != tskm_immediate_exec ) {
            KA_TRACE( 20, ( "__kmp_join_barrier: T#%d, old team = %d, old task_team = %p, th_task_team = %p\n",
                             __kmp_gtid_from_thread( this_thr ), team_id, team -> t.t_task_team,
                             this_thr->th.th_task_team ) );
            KMP_DEBUG_ASSERT( this_thr->th.th_task_team == team->t.t_task_team );
        }
        #endif /* KMP_DEBUG */
    #endif /* OMP_30_ENABLED */

    //
    // Copy the blocktime info to the thread, where __kmp_wait_sleep()
    // can access it when the team struct is not guaranteed to exist.
    //
    // Doing these loads causes a cache miss slows down EPCC parallel by 2x.
    // As a workaround, we do not perform the copy if blocktime=infinite,
    // since the values are not used by __kmp_wait_sleep() in that case.
    //
    if ( __kmp_dflt_blocktime != KMP_MAX_BLOCKTIME ) {
        #if OMP_30_ENABLED
            this_thr -> th.th_team_bt_intervals = team -> t.t_implicit_task_taskdata[tid].td_icvs.bt_intervals;
            this_thr -> th.th_team_bt_set = team -> t.t_implicit_task_taskdata[tid].td_icvs.bt_set;
        #else
            this_thr -> th.th_team_bt_intervals = team -> t.t_set_bt_intervals[tid];
            this_thr -> th.th_team_bt_set= team -> t.t_set_bt_set[tid];
        #endif // OMP_30_ENABLED
    }

#if USE_ITT_BUILD
    if ( __itt_sync_create_ptr || KMP_ITT_DEBUG )
        __kmp_itt_barrier_starting( gtid, itt_sync_obj );
#endif /* USE_ITT_BUILD */

    if ( __kmp_barrier_gather_pattern[ bs_forkjoin_barrier ] == bp_linear_bar || __kmp_barrier_gather_branch_bits[ bs_forkjoin_barrier ] == 0 ) {
        __kmp_linear_barrier_gather( bs_forkjoin_barrier, this_thr, gtid, tid, NULL
                                     USE_ITT_BUILD_ARG( itt_sync_obj )
                                     );
    } else if ( __kmp_barrier_gather_pattern[ bs_forkjoin_barrier ] == bp_tree_bar ) {
        __kmp_tree_barrier_gather( bs_forkjoin_barrier, this_thr, gtid, tid, NULL
                                   USE_ITT_BUILD_ARG( itt_sync_obj )
                                   );
    } else {
        __kmp_hyper_barrier_gather( bs_forkjoin_barrier, this_thr, gtid, tid, NULL
                                    USE_ITT_BUILD_ARG( itt_sync_obj )
                                    );
    }; // if

#if USE_ITT_BUILD
    if ( __itt_sync_create_ptr || KMP_ITT_DEBUG )
        __kmp_itt_barrier_middle( gtid, itt_sync_obj );
#endif /* USE_ITT_BUILD */

    //
    // From this point on, the team data structure may be deallocated
    // at any time by the master thread - it is unsafe to reference it
    // in any of the worker threads.
    //
    // Any per-team data items that need to be referenced before the end
    // of the barrier should be moved to the kmp_task_team_t structs.
    //

    #if OMP_30_ENABLED
        if ( KMP_MASTER_TID( tid ) ) {
            if ( __kmp_tasking_mode != tskm_immediate_exec ) {
                // Master shouldn't call decrease_load().         // TODO: enable master threads.
                // Master should have th_may_decrease_load == 0.  // TODO: enable master threads.
                __kmp_task_team_wait( this_thr, team
                                      USE_ITT_BUILD_ARG( itt_sync_obj )
                                      );
            }
#if USE_ITT_BUILD && USE_ITT_NOTIFY
            // Join barrier - report frame end
            if( __itt_frame_submit_v3_ptr && __kmp_forkjoin_frames_mode ) {
                kmp_uint64 tmp = __itt_get_timestamp();
                ident_t * loc = team->t.t_ident;
                switch( __kmp_forkjoin_frames_mode ) {
                case 1:
                  __kmp_itt_frame_submit( gtid, this_thr->th.th_frame_time, tmp, 0, loc );
                  break;
                case 2:
                  __kmp_itt_frame_submit( gtid, this_thr->th.th_bar_arrive_time, tmp, 1, loc );
                  break;
                case 3:
                  __kmp_itt_frame_submit( gtid, this_thr->th.th_frame_time, tmp, 0, loc );
                  __kmp_itt_frame_submit( gtid, this_thr->th.th_bar_arrive_time, tmp, 1, loc );
                  break;
                }
            }
#endif /* USE_ITT_BUILD */
        }
    #endif /* OMP_30_ENABLED */

    #if KMP_DEBUG
        if( KMP_MASTER_TID( tid )) {
            KA_TRACE( 15, ( "__kmp_join_barrier: T#%d(%d:%d) says all %d team threads arrived\n",
                            gtid, team_id, tid, nproc ));
        }
    #endif /* KMP_DEBUG */

    /* TODO now, mark worker threads as done so they may be disbanded */

    KMP_MB();       /* Flush all pending memory write invalidates.  */
    KA_TRACE( 10, ("__kmp_join_barrier: T#%d(%d:%d) leaving\n",
                   gtid, team_id, tid ));
}


/* TODO release worker threads' fork barriers as we are ready instead of all at once */

void
__kmp_fork_barrier( int gtid, int tid )
{
    kmp_info_t *this_thr = __kmp_threads[ gtid ];
    kmp_team_t *team     = ( tid == 0 ) ? this_thr -> th.th_team : NULL;
#if USE_ITT_BUILD
    void * itt_sync_obj = NULL;
#endif /* USE_ITT_BUILD */

    KA_TRACE( 10, ( "__kmp_fork_barrier: T#%d(%d:%d) has arrived\n",
                    gtid, ( team != NULL ) ? team->t.t_id : -1, tid ));

    /* th_team pointer only valid for master thread here */
    if ( KMP_MASTER_TID( tid ) ) {

#if USE_ITT_BUILD && USE_ITT_NOTIFY
            if ( __itt_sync_create_ptr || KMP_ITT_DEBUG ) {
                itt_sync_obj  = __kmp_itt_barrier_object( gtid, bs_forkjoin_barrier, 1 ); // create itt barrier object
                //__kmp_itt_barrier_starting( gtid, itt_sync_obj );   // AC: no need to call prepare right before acquired
                __kmp_itt_barrier_middle( gtid, itt_sync_obj );       // call acquired / releasing
            }
#endif /* USE_ITT_BUILD && USE_ITT_NOTIFY */

#ifdef KMP_DEBUG

        register kmp_info_t **other_threads = team -> t.t_threads;
        register int          i;

        /* verify state */
        KMP_MB();

        for( i = 1; i < team -> t.t_nproc ; i++ ) {
            KA_TRACE( 500, ( "__kmp_fork_barrier: T#%d(%d:0) checking T#%d(%d:%d) fork "
                             "go == %u.\n",
                             gtid, team->t.t_id, other_threads[i]->th.th_info.ds.ds_gtid,
                             team->t.t_id, other_threads[i]->th.th_info.ds.ds_tid,
                             other_threads[i]->th.th_bar[ bs_forkjoin_barrier ].bb.b_go ) );

            KMP_DEBUG_ASSERT( ( TCR_4( other_threads[i]->th.th_bar[bs_forkjoin_barrier].bb.b_go )
                                & ~(KMP_BARRIER_SLEEP_STATE) )
                               == KMP_INIT_BARRIER_STATE );
            KMP_DEBUG_ASSERT( other_threads[i]->th.th_team == team );

        }
#endif

#if OMP_30_ENABLED
        if ( __kmp_tasking_mode != tskm_immediate_exec ) {
            __kmp_task_team_setup( this_thr, team );
        }
#endif /* OMP_30_ENABLED */

        //
        // The master thread may have changed its blocktime between the
        // join barrier and the fork barrier.
        //
        // Copy the blocktime info to the thread, where __kmp_wait_sleep()
        // can access it when the team struct is not guaranteed to exist.
        //
        // See the note about the corresponding code in __kmp_join_barrier()
        // being performance-critical.
        //
        if ( __kmp_dflt_blocktime != KMP_MAX_BLOCKTIME ) {
#if OMP_30_ENABLED
            this_thr -> th.th_team_bt_intervals = team -> t.t_implicit_task_taskdata[tid].td_icvs.bt_intervals;
            this_thr -> th.th_team_bt_set = team -> t.t_implicit_task_taskdata[tid].td_icvs.bt_set;
#else
            this_thr -> th.th_team_bt_intervals = team -> t.t_set_bt_intervals[tid];
            this_thr -> th.th_team_bt_set= team -> t.t_set_bt_set[tid];
#endif // OMP_30_ENABLED
        }
    } // master

    if ( __kmp_barrier_release_pattern[ bs_forkjoin_barrier ] == bp_linear_bar || __kmp_barrier_release_branch_bits[ bs_forkjoin_barrier ] == 0 ) {
        __kmp_linear_barrier_release( bs_forkjoin_barrier, this_thr, gtid, tid, TRUE
                                      USE_ITT_BUILD_ARG( itt_sync_obj )
                                      );
    } else if ( __kmp_barrier_release_pattern[ bs_forkjoin_barrier ] == bp_tree_bar ) {
        __kmp_tree_barrier_release( bs_forkjoin_barrier, this_thr, gtid, tid, TRUE
                                    USE_ITT_BUILD_ARG( itt_sync_obj )
                                    );
    } else {
        __kmp_hyper_barrier_release( bs_forkjoin_barrier, this_thr, gtid, tid, TRUE
                                     USE_ITT_BUILD_ARG( itt_sync_obj )
                                     );
    }; // if

    //
    // early exit for reaping threads releasing forkjoin barrier
    //
    if ( TCR_4(__kmp_global.g.g_done) ) {

#if OMP_30_ENABLED
        if ( this_thr->th.th_task_team != NULL ) {
            if ( KMP_MASTER_TID( tid ) ) {
                TCW_PTR(this_thr->th.th_task_team, NULL);
            }
            else {
                __kmp_unref_task_team( this_thr->th.th_task_team, this_thr );
            }
        }
#endif /* OMP_30_ENABLED */

#if USE_ITT_BUILD && USE_ITT_NOTIFY
        if ( __itt_sync_create_ptr || KMP_ITT_DEBUG ) {
            if ( !KMP_MASTER_TID( tid ) ) {
                itt_sync_obj  = __kmp_itt_barrier_object( gtid, bs_forkjoin_barrier );
                if ( itt_sync_obj )
                    __kmp_itt_barrier_finished( gtid, itt_sync_obj );
            }
        }
#endif /* USE_ITT_BUILD && USE_ITT_NOTIFY */
        KA_TRACE( 10, ( "__kmp_fork_barrier: T#%d is leaving early\n", gtid ));
        return;
    }

    //
    // We can now assume that a valid team structure has been allocated
    // by the master and propagated to all worker threads.
    //
    // The current thread, however, may not be part of the team, so we can't
    // blindly assume that the team pointer is non-null.
    //
    team = (kmp_team_t *)TCR_PTR(this_thr->th.th_team);
    KMP_DEBUG_ASSERT( team != NULL );
    tid = __kmp_tid_from_gtid( gtid );

#if OMP_30_ENABLED

# if KMP_BARRIER_ICV_PULL
    // Master thread's copy of the ICVs was set up on the implicit taskdata in __kmp_reinitialize_team.
    // __kmp_fork_call() assumes the master thread's implicit task has this data before this function is called.
    // We cannot modify __kmp_fork_call() to look at the fixed ICVs in the master's thread struct, because it is
    // not always the case that the threads arrays have been allocated when __kmp_fork_call() is executed.
    if (! KMP_MASTER_TID( tid ) ) {  // master thread already has ICVs
        // Copy the initial ICVs from the master's thread struct to the implicit task for this tid.
        KA_TRACE( 10, ( "__kmp_fork_barrier: T#%d(%d) is PULLing ICVs\n", gtid, tid ));
        load_icvs(&team->t.t_threads[0]->th.th_fixed_icvs);
        __kmp_init_implicit_task( team->t.t_ident, team->t.t_threads[tid], team, tid, FALSE );
        store_icvs(&team->t.t_implicit_task_taskdata[tid].td_icvs, &team->t.t_threads[0]->th.th_fixed_icvs);
        sync_icvs();
    }
# endif // KMP_BARRIER_ICV_PULL

    if ( __kmp_tasking_mode != tskm_immediate_exec ) {
        __kmp_task_team_sync( this_thr, team );
    }

#endif /* OMP_30_ENABLED */

#if OMP_40_ENABLED && KMP_AFFINITY_SUPPORTED
    kmp_proc_bind_t proc_bind = team->t.t_proc_bind;
    if ( proc_bind == proc_bind_intel ) {
#endif
#if KMP_MIC
        //
        // Call dynamic affinity settings
        //
        if( __kmp_affinity_type == affinity_balanced && team->t.t_size_changed ) {
            __kmp_balanced_affinity( tid, team->t.t_nproc );
        }
#endif
#if OMP_40_ENABLED && KMP_AFFINITY_SUPPORTED
    }
    else if ( ( proc_bind != proc_bind_false )
              && ( proc_bind != proc_bind_disabled )) {
        if ( this_thr->th.th_new_place == this_thr->th.th_current_place ) {
            KA_TRACE( 100, ( "__kmp_fork_barrier: T#%d already in correct place %d\n",
                             __kmp_gtid_from_thread( this_thr ), this_thr->th.th_current_place ) );
        }
        else {
            __kmp_affinity_set_place( gtid );
        }
    }
#endif

#if USE_ITT_BUILD && USE_ITT_NOTIFY
    if ( __itt_sync_create_ptr || KMP_ITT_DEBUG ) {
        if ( !KMP_MASTER_TID( tid ) ) {
            itt_sync_obj  = __kmp_itt_barrier_object( gtid, bs_forkjoin_barrier ); // get correct barrier object
            __kmp_itt_barrier_finished( gtid, itt_sync_obj );   // workers call acquired
        }                                                                          // (prepare called inside barrier_release)
    }
#endif /* USE_ITT_BUILD && USE_ITT_NOTIFY */
    KA_TRACE( 10, ( "__kmp_fork_barrier: T#%d(%d:%d) is leaving\n",
      gtid, team->t.t_id, tid ));
}


/* ------------------------------------------------------------------------ */
/* ------------------------------------------------------------------------ */

void *
__kmp_launch_thread( kmp_info_t *this_thr )
{
    int                   gtid = this_thr->th.th_info.ds.ds_gtid;
/*    void                 *stack_data;*/
    kmp_team_t *(*volatile pteam);

    KMP_MB();
    KA_TRACE( 10, ("__kmp_launch_thread: T#%d start\n", gtid ) );

    if( __kmp_env_consistency_check ) {
        this_thr -> th.th_cons = __kmp_allocate_cons_stack( gtid );  // ATT: Memory leak?
    }

    /* This is the place where threads wait for work */
    while( ! TCR_4(__kmp_global.g.g_done) ) {
        KMP_DEBUG_ASSERT( this_thr == __kmp_threads[ gtid ] );
        KMP_MB();

        /* wait for work to do */
        KA_TRACE( 20, ("__kmp_launch_thread: T#%d waiting for work\n", gtid ));

        /* No tid yet since not part of a team */
        __kmp_fork_barrier( gtid, KMP_GTID_DNE );

        pteam = (kmp_team_t *(*))(& this_thr->th.th_team);

        /* have we been allocated? */
        if ( TCR_SYNC_PTR(*pteam) && !TCR_4(__kmp_global.g.g_done) ) {
            /* we were just woken up, so run our new task */
            if ( TCR_SYNC_PTR((*pteam)->t.t_pkfn) != NULL ) {
                int rc;
                KA_TRACE( 20, ("__kmp_launch_thread: T#%d(%d:%d) invoke microtask = %p\n",
                    gtid, (*pteam)->t.t_id, __kmp_tid_from_gtid(gtid), (*pteam)->t.t_pkfn ));

#if KMP_ARCH_X86 || KMP_ARCH_X86_64
                if ( __kmp_inherit_fp_control && (*pteam)->t.t_fp_control_saved ) {
                    __kmp_clear_x87_fpu_status_word();
                    __kmp_load_x87_fpu_control_word( &(*pteam)->t.t_x87_fpu_control_word );
                    __kmp_load_mxcsr( &(*pteam)->t.t_mxcsr );
                }
#endif /* KMP_ARCH_X86 || KMP_ARCH_X86_64 */

                rc = (*pteam) -> t.t_invoke( gtid );
                KMP_ASSERT( rc );

                KMP_MB();
                KA_TRACE( 20, ("__kmp_launch_thread: T#%d(%d:%d) done microtask = %p\n",
                        gtid, (*pteam)->t.t_id, __kmp_tid_from_gtid(gtid), (*pteam)->t.t_pkfn ));
            }

            /* join barrier after parallel region */
            __kmp_join_barrier( gtid );
        }
    }
    TCR_SYNC_PTR(__kmp_global.g.g_done);

#if OMP_30_ENABLED
    if ( TCR_PTR( this_thr->th.th_task_team ) != NULL ) {
        __kmp_unref_task_team( this_thr->th.th_task_team, this_thr );
    }
#endif /* OMP_30_ENABLED */

    /* run the destructors for the threadprivate data for this thread */
    __kmp_common_destroy_gtid( gtid );

    KA_TRACE( 10, ("__kmp_launch_thread: T#%d done\n", gtid ) );
    KMP_MB();
    return this_thr;
}

/* ------------------------------------------------------------------------ */
/* ------------------------------------------------------------------------ */



void
__kmp_internal_end_dest( void *specific_gtid )
{
    #if KMP_COMPILER_ICC
        #pragma warning( push )
        #pragma warning( disable:  810 ) // conversion from "void *" to "int" may lose significant bits
    #endif
    // Make sure no significant bits are lost
    int gtid = (kmp_intptr_t)specific_gtid - 1;
    #if KMP_COMPILER_ICC
        #pragma warning( pop )
    #endif

    KA_TRACE( 30, ("__kmp_internal_end_dest: T#%d\n", gtid));
    /* NOTE: the gtid is stored as gitd+1 in the thread-local-storage
     * this is because 0 is reserved for the nothing-stored case */

    /* josh: One reason for setting the gtid specific data even when it is being
       destroyed by pthread is to allow gtid lookup through thread specific data
       (__kmp_gtid_get_specific).  Some of the code, especially stat code,
       that gets executed in the call to __kmp_internal_end_thread, actually
       gets the gtid through the thread specific data.  Setting it here seems
       rather inelegant and perhaps wrong, but allows __kmp_internal_end_thread
       to run smoothly.
       todo: get rid of this after we remove the dependence on
       __kmp_gtid_get_specific
    */
    if(gtid >= 0 && KMP_UBER_GTID(gtid))
        __kmp_gtid_set_specific( gtid );
    #ifdef KMP_TDATA_GTID
        __kmp_gtid = gtid;
    #endif
    __kmp_internal_end_thread( gtid );
}

#if KMP_OS_UNIX && GUIDEDLL_EXPORTS

// 2009-09-08 (lev): It looks the destructor does not work. In simple test cases destructors work
// perfectly, but in real libiomp5.so I have no evidence it is ever called. However, -fini linker
// option in makefile.mk works fine.

__attribute__(( destructor ))
void
__kmp_internal_end_dtor( void )
{
    __kmp_internal_end_atexit();
}

void
__kmp_internal_end_fini( void )
{
    __kmp_internal_end_atexit();
}

#endif

/* [Windows] josh: when the atexit handler is called, there may still be more than one thread alive */
void
__kmp_internal_end_atexit( void )
{
    KA_TRACE( 30, ( "__kmp_internal_end_atexit\n" ) );
    /* [Windows]
       josh: ideally, we want to completely shutdown the library in this atexit handler, but
       stat code that depends on thread specific data for gtid fails because that data becomes
       unavailable at some point during the shutdown, so we call __kmp_internal_end_thread
       instead.  We should eventually remove the dependency on __kmp_get_specific_gtid in the
       stat code and use __kmp_internal_end_library to cleanly shutdown the library.

// TODO: Can some of this comment about GVS be removed?
       I suspect that the offending stat code is executed when the calling thread tries to
       clean up a dead root thread's data structures, resulting in GVS code trying to close
       the GVS structures for that thread, but since the stat code uses
       __kmp_get_specific_gtid to get the gtid with the assumption that the calling thread is
       cleaning up itself instead of another thread, it gets confused.  This happens because
       allowing a thread to unregister and cleanup another thread is a recent modification for
       addressing an issue with Maxon Cinema4D.  Based on the current design (20050722), a
       thread may end up trying to unregister another thread only if thread death does not
       trigger the calling of __kmp_internal_end_thread.  For Linux* OS, there is the thread
       specific data destructor function to detect thread death.  For Windows dynamic, there
       is DllMain(THREAD_DETACH).  For Windows static, there is nothing.  Thus, the
       workaround is applicable only for Windows static stat library.
    */
    __kmp_internal_end_library( -1 );
    #if KMP_OS_WINDOWS
        __kmp_close_console();
    #endif
}

static void
__kmp_reap_thread(
    kmp_info_t * thread,
    int is_root
) {

    // It is assumed __kmp_forkjoin_lock is acquired.

    int gtid;

    KMP_DEBUG_ASSERT( thread != NULL );

    gtid = thread->th.th_info.ds.ds_gtid;

    if ( ! is_root ) {

        if ( __kmp_dflt_blocktime != KMP_MAX_BLOCKTIME ) {
            /* Assume the threads are at the fork barrier here */
            KA_TRACE( 20, ("__kmp_reap_thread: releasing T#%d from fork barrier for reap\n", gtid ) );
            /* Need release fence here to prevent seg faults for tree forkjoin barrier (GEH) */
            __kmp_release(
                thread,
                &thread->th.th_bar[ bs_forkjoin_barrier ].bb.b_go,
                kmp_release_fence
            );
        }; // if


        // Terminate OS thread.
        __kmp_reap_worker( thread );

        //
        // The thread was killed asynchronously.  If it was actively
        // spinning in the in the thread pool, decrement the global count.
        //
        // There is a small timing hole here - if the worker thread was
        // just waking up after sleeping in the pool, had reset it's
        // th_active_in_pool flag but not decremented the global counter
        // __kmp_thread_pool_active_nth yet, then the global counter
        // might not get updated.
        //
        // Currently, this can only happen as the library is unloaded,
        // so there are no harmful side effects.
        //
        if ( thread->th.th_active_in_pool ) {
            thread->th.th_active_in_pool = FALSE;
            KMP_TEST_THEN_DEC32(
              (kmp_int32 *) &__kmp_thread_pool_active_nth );
            KMP_DEBUG_ASSERT( TCR_4(__kmp_thread_pool_active_nth) >= 0 );
        }

        // Decrement # of [worker] threads in the pool.
        KMP_DEBUG_ASSERT( __kmp_thread_pool_nth > 0 );
        --__kmp_thread_pool_nth;
    }; // if

    // Free the fast memory for tasking
    #if USE_FAST_MEMORY
        __kmp_free_fast_memory( thread );
    #endif /* USE_FAST_MEMORY */

    __kmp_suspend_uninitialize_thread( thread );

    KMP_DEBUG_ASSERT( __kmp_threads[ gtid ] == thread );
    TCW_SYNC_PTR(__kmp_threads[gtid], NULL);

    -- __kmp_all_nth;
    // __kmp_nth was decremented when thread is added to the pool.

#ifdef KMP_ADJUST_BLOCKTIME
    /* Adjust blocktime back to user setting or default if necessary */
    /* Middle initialization might never have occurred                */
    if ( !__kmp_env_blocktime && ( __kmp_avail_proc > 0 ) ) {
        KMP_DEBUG_ASSERT( __kmp_avail_proc > 0 );
        if ( __kmp_nth <= __kmp_avail_proc ) {
            __kmp_zero_bt = FALSE;
        }
    }
#endif /* KMP_ADJUST_BLOCKTIME */

    /* free the memory being used */
    if( __kmp_env_consistency_check ) {
        if ( thread->th.th_cons ) {
            __kmp_free_cons_stack( thread->th.th_cons );
            thread->th.th_cons = NULL;
        }; // if
    }

    if ( thread->th.th_pri_common != NULL ) {
        __kmp_free( thread->th.th_pri_common );
        thread->th.th_pri_common = NULL;
    }; // if

    #if KMP_USE_BGET
        if ( thread->th.th_local.bget_data != NULL ) {
            __kmp_finalize_bget( thread );
        }; // if
    #endif

#if KMP_AFFINITY_SUPPORTED
    if ( thread->th.th_affin_mask != NULL ) {
        KMP_CPU_FREE( thread->th.th_affin_mask );
        thread->th.th_affin_mask = NULL;
    }; // if
#endif /* KMP_AFFINITY_SUPPORTED */

    __kmp_reap_team( thread->th.th_serial_team );
    thread->th.th_serial_team = NULL;
    __kmp_free( thread );

    KMP_MB();

} // __kmp_reap_thread

static void
__kmp_internal_end(void)
{
    int i;

    /* First, unregister the library */
    __kmp_unregister_library();

    #if KMP_OS_WINDOWS
        /* In Win static library, we can't tell when a root actually dies, so we
           reclaim the data structures for any root threads that have died but not
           unregistered themselves, in order to shut down cleanly.
           In Win dynamic library we also can't tell when a thread dies.
        */
        __kmp_reclaim_dead_roots(); // AC: moved here to always clean resources of dead roots
    #endif

    for( i=0 ; i<__kmp_threads_capacity ; i++ )
        if( __kmp_root[i] )
            if( __kmp_root[i] -> r.r_active )
                break;
    KMP_MB();       /* Flush all pending memory write invalidates.  */
    TCW_SYNC_4(__kmp_global.g.g_done, TRUE);

    if ( i < __kmp_threads_capacity ) {
        // 2009-09-08 (lev): Other alive roots found. Why do we kill the monitor??
        KMP_MB();       /* Flush all pending memory write invalidates.  */

        //
        // Need to check that monitor was initialized before reaping it.
        // If we are called form __kmp_atfork_child (which sets
        // __kmp_init_parallel = 0), then __kmp_monitor will appear to
        // contain valid data, but it is only valid in the parent process,
        // not the child.
        //
        // One of the possible fixes for CQ138434 / CQ140126
        // (used in 20091103_dreamworks patch)
        //
        // New behavior (201008): instead of keying off of the flag
        // __kmp_init_parallel, the monitor thread creation is keyed off
        // of the new flag __kmp_init_monitor.
        //
        __kmp_acquire_bootstrap_lock( & __kmp_monitor_lock );
        if ( TCR_4( __kmp_init_monitor ) ) {
            __kmp_reap_monitor( & __kmp_monitor );
            TCW_4( __kmp_init_monitor, 0 );
        }
        __kmp_release_bootstrap_lock( & __kmp_monitor_lock );
        KA_TRACE( 10, ("__kmp_internal_end: monitor reaped\n" ) );
    } else {
        /* TODO move this to cleanup code */
        #ifdef KMP_DEBUG
            /* make sure that everything has properly ended */
            for ( i = 0; i < __kmp_threads_capacity; i++ ) {
                if( __kmp_root[i] ) {
                    KMP_ASSERT( ! KMP_UBER_GTID( i ) );
                    KMP_ASSERT( ! __kmp_root[i] -> r.r_active );
                }
            }
        #endif

        KMP_MB();

        // Reap the worker threads.
        // This is valid for now, but be careful if threads are reaped sooner.
        while ( __kmp_thread_pool != NULL ) {    // Loop thru all the thread in the pool.
            // Get the next thread from the pool.
            kmp_info_t * thread = (kmp_info_t *) __kmp_thread_pool;
            __kmp_thread_pool = thread->th.th_next_pool;
            // Reap it.
            thread->th.th_next_pool = NULL;
            thread->th.th_in_pool = FALSE;
            __kmp_reap_thread( thread, 0 );
        }; // while
        __kmp_thread_pool_insert_pt = NULL;

        // Reap teams.
        while ( __kmp_team_pool != NULL ) {     // Loop thru all the teams in the pool.
            // Get the next team from the pool.
            kmp_team_t * team = (kmp_team_t *) __kmp_team_pool;
            __kmp_team_pool = team->t.t_next_pool;
            // Reap it.
            team->t.t_next_pool = NULL;
            __kmp_reap_team( team );
        }; // while

        #if OMP_30_ENABLED
            __kmp_reap_task_teams( );
        #endif /* OMP_30_ENABLED */

        for ( i = 0; i < __kmp_threads_capacity; ++ i ) {
            // TBD: Add some checking...
            // Something like KMP_DEBUG_ASSERT( __kmp_thread[ i ] == NULL );
        }

        /* Make sure all threadprivate destructors get run by joining with all worker
           threads before resetting this flag */
        TCW_SYNC_4(__kmp_init_common, FALSE);

        KA_TRACE( 10, ("__kmp_internal_end: all workers reaped\n" ) );
        KMP_MB();

        //
        // See note above: One of the possible fixes for CQ138434 / CQ140126
        //
        // FIXME: push both code fragments down and CSE them?
        // push them into __kmp_cleanup() ?
        //
        __kmp_acquire_bootstrap_lock( & __kmp_monitor_lock );
        if ( TCR_4( __kmp_init_monitor ) ) {
            __kmp_reap_monitor( & __kmp_monitor );
            TCW_4( __kmp_init_monitor, 0 );
        }
        __kmp_release_bootstrap_lock( & __kmp_monitor_lock );
        KA_TRACE( 10, ("__kmp_internal_end: monitor reaped\n" ) );

    } /* else !__kmp_global.t_active */
    TCW_4(__kmp_init_gtid, FALSE);
    KMP_MB();       /* Flush all pending memory write invalidates.  */


    __kmp_cleanup();
}

void
__kmp_internal_end_library( int gtid_req )
{
    int i;

    /* if we have already cleaned up, don't try again, it wouldn't be pretty */
    /* this shouldn't be a race condition because __kmp_internal_end() is the
     * only place to clear __kmp_serial_init */
    /* we'll check this later too, after we get the lock */
    // 2009-09-06: We do not set g_abort without setting g_done. This check looks redundaant,
    // because the next check will work in any case.
    if( __kmp_global.g.g_abort ) {
        KA_TRACE( 11, ("__kmp_internal_end_library: abort, exiting\n" ));
        /* TODO abort? */
        return;
    }
    if( TCR_4(__kmp_global.g.g_done) || !__kmp_init_serial ) {
        KA_TRACE( 10, ("__kmp_internal_end_library: already finished\n" ));
        return;
    }


    KMP_MB();       /* Flush all pending memory write invalidates.  */

    /* find out who we are and what we should do */
    {
        int gtid = (gtid_req>=0) ? gtid_req : __kmp_gtid_get_specific();
        KA_TRACE( 10, ("__kmp_internal_end_library: enter T#%d  (%d)\n", gtid, gtid_req ));
        if( gtid == KMP_GTID_SHUTDOWN ) {
            KA_TRACE( 10, ("__kmp_internal_end_library: !__kmp_init_runtime, system already shutdown\n" ));
            return;
        } else if( gtid == KMP_GTID_MONITOR ) {
            KA_TRACE( 10, ("__kmp_internal_end_library: monitor thread, gtid not registered, or system shutdown\n" ));
            return;
        } else if( gtid == KMP_GTID_DNE ) {
            KA_TRACE( 10, ("__kmp_internal_end_library: gtid not registered or system shutdown\n" ));
            /* we don't know who we are, but we may still shutdown the library */
        } else if( KMP_UBER_GTID( gtid )) {
            /* unregister ourselves as an uber thread.  gtid is no longer valid */
            if( __kmp_root[gtid] -> r.r_active ) {
                __kmp_global.g.g_abort = -1;
                TCW_SYNC_4(__kmp_global.g.g_done, TRUE);
                KA_TRACE( 10, ("__kmp_internal_end_library: root still active, abort T#%d\n", gtid ));
                return;
            } else {
                KA_TRACE( 10, ("__kmp_internal_end_library: unregistering sibling T#%d\n", gtid ));
                __kmp_unregister_root_current_thread( gtid );
            }
        } else {
            /* worker threads may call this function through the atexit handler, if they call exit() */
            /* For now, skip the usual subsequent processing and just dump the debug buffer.
               TODO: do a thorough shutdown instead
            */
            #ifdef DUMP_DEBUG_ON_EXIT
                if ( __kmp_debug_buf )
                    __kmp_dump_debug_buffer( );
            #endif
            return;
        }
    }
    /* synchronize the termination process */
    __kmp_acquire_bootstrap_lock( &__kmp_initz_lock );

    /* have we already finished */
    if( __kmp_global.g.g_abort ) {
        KA_TRACE( 10, ("__kmp_internal_end_library: abort, exiting\n" ));
        /* TODO abort? */
        __kmp_release_bootstrap_lock( &__kmp_initz_lock );
        return;
    }
    if( TCR_4(__kmp_global.g.g_done) || !__kmp_init_serial ) {
        __kmp_release_bootstrap_lock( &__kmp_initz_lock );
        return;
    }

    /* We need this lock to enforce mutex between this reading of
       __kmp_threads_capacity and the writing by __kmp_register_root.
       Alternatively, we can use a counter of roots that is
       atomically updated by __kmp_get_global_thread_id_reg,
       __kmp_do_serial_initialize and __kmp_internal_end_*.
    */
    __kmp_acquire_bootstrap_lock( &__kmp_forkjoin_lock );

    /* now we can safely conduct the actual termination */
    __kmp_internal_end();

    __kmp_release_bootstrap_lock( &__kmp_forkjoin_lock );
    __kmp_release_bootstrap_lock( &__kmp_initz_lock );

    KA_TRACE( 10, ("__kmp_internal_end_library: exit\n" ) );

    #ifdef DUMP_DEBUG_ON_EXIT
        if ( __kmp_debug_buf )
            __kmp_dump_debug_buffer();
    #endif

    #if KMP_OS_WINDOWS
        __kmp_close_console();
    #endif

    __kmp_fini_allocator();

} // __kmp_internal_end_library

void
__kmp_internal_end_thread( int gtid_req )
{
    int i;

    /* if we have already cleaned up, don't try again, it wouldn't be pretty */
    /* this shouldn't be a race condition because __kmp_internal_end() is the
     * only place to clear __kmp_serial_init */
    /* we'll check this later too, after we get the lock */
    // 2009-09-06: We do not set g_abort without setting g_done. This check looks redundant,
    // because the next check will work in any case.
    if( __kmp_global.g.g_abort ) {
        KA_TRACE( 11, ("__kmp_internal_end_thread: abort, exiting\n" ));
        /* TODO abort? */
        return;
    }
    if( TCR_4(__kmp_global.g.g_done) || !__kmp_init_serial ) {
        KA_TRACE( 10, ("__kmp_internal_end_thread: already finished\n" ));
        return;
    }

    KMP_MB();       /* Flush all pending memory write invalidates.  */

    /* find out who we are and what we should do */
    {
        int gtid = (gtid_req>=0) ? gtid_req : __kmp_gtid_get_specific();
        KA_TRACE( 10, ("__kmp_internal_end_thread: enter T#%d  (%d)\n", gtid, gtid_req ));
        if( gtid == KMP_GTID_SHUTDOWN ) {
            KA_TRACE( 10, ("__kmp_internal_end_thread: !__kmp_init_runtime, system already shutdown\n" ));
            return;
        } else if( gtid == KMP_GTID_MONITOR ) {
            KA_TRACE( 10, ("__kmp_internal_end_thread: monitor thread, gtid not registered, or system shutdown\n" ));
            return;
        } else if( gtid == KMP_GTID_DNE ) {
            KA_TRACE( 10, ("__kmp_internal_end_thread: gtid not registered or system shutdown\n" ));
            return;
            /* we don't know who we are */
        } else if( KMP_UBER_GTID( gtid )) {
        /* unregister ourselves as an uber thread.  gtid is no longer valid */
            if( __kmp_root[gtid] -> r.r_active ) {
                __kmp_global.g.g_abort = -1;
                TCW_SYNC_4(__kmp_global.g.g_done, TRUE);
                KA_TRACE( 10, ("__kmp_internal_end_thread: root still active, abort T#%d\n", gtid ));
                return;
            } else {
                KA_TRACE( 10, ("__kmp_internal_end_thread: unregistering sibling T#%d\n", gtid ));
                __kmp_unregister_root_current_thread( gtid );
            }
        } else {
            /* just a worker thread, let's leave */
            KA_TRACE( 10, ("__kmp_internal_end_thread: worker thread T#%d\n", gtid ));

            #if OMP_30_ENABLED
                if ( gtid >= 0 ) {
                    kmp_info_t *this_thr = __kmp_threads[ gtid ];
                    if (TCR_PTR(this_thr->th.th_task_team) != NULL) {
                        __kmp_unref_task_team(this_thr->th.th_task_team, this_thr);
                    }
                }
            #endif /* OMP_30_ENABLED */

            KA_TRACE( 10, ("__kmp_internal_end_thread: worker thread done, exiting T#%d\n", gtid ));
            return;
        }
    }
    #if defined GUIDEDLL_EXPORTS
    // AC: lets not shutdown the Linux* OS dynamic library at the exit of uber thread,
    //     because we will better shutdown later in the library destructor.
    //     The reason of this change is performance problem when non-openmp thread
    //     in a loop forks and joins many openmp threads. We can save a lot of time
    //     keeping worker threads alive until the program shutdown.
    // OM: Removed Linux* OS restriction to fix the crash on OS X* (DPD200239966) and
    //     Windows(DPD200287443) that occurs when using critical sections from foreign threads.
        KA_TRACE( 10, ("__kmp_internal_end_thread: exiting\n") );
        return;
    #endif
    /* synchronize the termination process */
    __kmp_acquire_bootstrap_lock( &__kmp_initz_lock );

    /* have we already finished */
    if( __kmp_global.g.g_abort ) {
        KA_TRACE( 10, ("__kmp_internal_end_thread: abort, exiting\n" ));
        /* TODO abort? */
        __kmp_release_bootstrap_lock( &__kmp_initz_lock );
        return;
    }
    if( TCR_4(__kmp_global.g.g_done) || !__kmp_init_serial ) {
        __kmp_release_bootstrap_lock( &__kmp_initz_lock );
        return;
    }

    /* We need this lock to enforce mutex between this reading of
       __kmp_threads_capacity and the writing by __kmp_register_root.
       Alternatively, we can use a counter of roots that is
       atomically updated by __kmp_get_global_thread_id_reg,
       __kmp_do_serial_initialize and __kmp_internal_end_*.
    */

    /* should we finish the run-time?  are all siblings done? */
    __kmp_acquire_bootstrap_lock( &__kmp_forkjoin_lock );

    for ( i = 0; i < __kmp_threads_capacity; ++ i ) {
        if ( KMP_UBER_GTID( i ) ) {
            KA_TRACE( 10, ("__kmp_internal_end_thread: remaining sibling task: gtid==%d\n", i ));
            __kmp_release_bootstrap_lock( &__kmp_forkjoin_lock );
            __kmp_release_bootstrap_lock( &__kmp_initz_lock );
            return;
        };
    }

    /* now we can safely conduct the actual termination */

    __kmp_internal_end();

    __kmp_release_bootstrap_lock( &__kmp_forkjoin_lock );
    __kmp_release_bootstrap_lock( &__kmp_initz_lock );

    KA_TRACE( 10, ("__kmp_internal_end_thread: exit\n" ) );

    #ifdef DUMP_DEBUG_ON_EXIT
        if ( __kmp_debug_buf )
            __kmp_dump_debug_buffer();
    #endif
} // __kmp_internal_end_thread

// -------------------------------------------------------------------------------------------------
// Library registration stuff.

static long   __kmp_registration_flag = 0;
    // Random value used to indicate library initialization.
static char * __kmp_registration_str  = NULL;
    // Value to be saved in env var __KMP_REGISTERED_LIB_<pid>.


static inline
char *
__kmp_reg_status_name() {
    /*
        On RHEL 3u5 if linked statically, getpid() returns different values in each thread.
        If registration and unregistration go in different threads (omp_misc_other_root_exit.cpp test case),
        the name of registered_lib_env env var can not be found, because the name will contain different pid.
    */
    return __kmp_str_format( "__KMP_REGISTERED_LIB_%d", (int) getpid() );
} // __kmp_reg_status_get


void
__kmp_register_library_startup(
    void
) {

    char * name   = __kmp_reg_status_name();  // Name of the environment variable.
    int    done   = 0;
    union {
        double dtime;
        long   ltime;
    } time;
    #if KMP_OS_WINDOWS
        __kmp_initialize_system_tick();
    #endif
    __kmp_read_system_time( & time.dtime );
    __kmp_registration_flag = 0xCAFE0000L | ( time.ltime & 0x0000FFFFL );
    __kmp_registration_str =
        __kmp_str_format(
            "%p-%lx-%s",
            & __kmp_registration_flag,
            __kmp_registration_flag,
            KMP_LIBRARY_FILE
        );

    KA_TRACE( 50, ( "__kmp_register_library_startup: %s=\"%s\"\n", name, __kmp_registration_str ) );

    while ( ! done ) {

        char * value  = NULL; // Actual value of the environment variable.

        // Set environment variable, but do not overwrite if it is exist.
        __kmp_env_set( name, __kmp_registration_str, 0 );
        // Check the variable is written.
        value = __kmp_env_get( name );
        if ( value != NULL && strcmp( value, __kmp_registration_str ) == 0 ) {

            done = 1;    // Ok, environment variable set successfully, exit the loop.

        } else {

            // Oops. Write failed. Another copy of OpenMP RTL is in memory.
            // Check whether it alive or dead.
            int    neighbor = 0; // 0 -- unknown status, 1 -- alive, 2 -- dead.
            char * tail          = value;
            char * flag_addr_str = NULL;
            char * flag_val_str  = NULL;
            char const * file_name     = NULL;
            __kmp_str_split( tail, '-', & flag_addr_str, & tail );
            __kmp_str_split( tail, '-', & flag_val_str,  & tail );
            file_name = tail;
            if ( tail != NULL ) {
                long * flag_addr = 0;
                long   flag_val  = 0;
                sscanf( flag_addr_str, "%p",  & flag_addr );
                sscanf( flag_val_str,  "%lx", & flag_val  );
                if ( flag_addr != 0 && flag_val != 0 && strcmp( file_name, "" ) != 0 ) {
                    // First, check whether environment-encoded address is mapped into addr space.
                    // If so, dereference it to see if it still has the right value.

                    if ( __kmp_is_address_mapped( flag_addr ) && * flag_addr == flag_val ) {
                        neighbor = 1;
                    } else {
                        // If not, then we know the other copy of the library is no longer running.
                        neighbor = 2;
                    }; // if
                }; // if
            }; // if
            switch ( neighbor ) {
                case 0 :      // Cannot parse environment variable -- neighbor status unknown.
                    // Assume it is the incompatible format of future version of the library.
                    // Assume the other library is alive.
                    // WARN( ... ); // TODO: Issue a warning.
                    file_name = "unknown library";
                    // Attention! Falling to the next case. That's intentional.
                case 1 : {    // Neighbor is alive.
                    // Check it is allowed.
                    char * duplicate_ok = __kmp_env_get( "KMP_DUPLICATE_LIB_OK" );
                    if ( ! __kmp_str_match_true( duplicate_ok ) ) {
                        // That's not allowed. Issue fatal error.
                        __kmp_msg(
                            kmp_ms_fatal,
                            KMP_MSG( DuplicateLibrary, KMP_LIBRARY_FILE, file_name ),
                            KMP_HNT( DuplicateLibrary ),
                            __kmp_msg_null
                        );
                    }; // if
                    KMP_INTERNAL_FREE( duplicate_ok );
                    __kmp_duplicate_library_ok = 1;
                    done = 1;    // Exit the loop.
                } break;
                case 2 : {    // Neighbor is dead.
                    // Clear the variable and try to register library again.
                    __kmp_env_unset( name );
                }  break;
                default : {
                    KMP_DEBUG_ASSERT( 0 );
                } break;
            }; // switch

        }; // if
        KMP_INTERNAL_FREE( (void *) value );

    }; // while
    KMP_INTERNAL_FREE( (void *) name );

} // func __kmp_register_library_startup


void
__kmp_unregister_library( void ) {

    char * name  = __kmp_reg_status_name();
    char * value = __kmp_env_get( name );

    KMP_DEBUG_ASSERT( __kmp_registration_flag != 0 );
    KMP_DEBUG_ASSERT( __kmp_registration_str  != NULL );
    if ( value != NULL && strcmp( value, __kmp_registration_str ) == 0 ) {
        // Ok, this is our variable. Delete it.
        __kmp_env_unset( name );
    }; // if

    KMP_INTERNAL_FREE( __kmp_registration_str );
    KMP_INTERNAL_FREE( value );
    KMP_INTERNAL_FREE( name );

    __kmp_registration_flag = 0;
    __kmp_registration_str  = NULL;

} // __kmp_unregister_library


// End of Library registration stuff.
// -------------------------------------------------------------------------------------------------

static void
__kmp_do_serial_initialize( void )
{
    int i, gtid;
    int size;

    KA_TRACE( 10, ("__kmp_serial_initialize: enter\n" ) );

    KMP_DEBUG_ASSERT( sizeof( kmp_int32 ) == 4 );
    KMP_DEBUG_ASSERT( sizeof( kmp_uint32 ) == 4 );
    KMP_DEBUG_ASSERT( sizeof( kmp_int64 ) == 8 );
    KMP_DEBUG_ASSERT( sizeof( kmp_uint64 ) == 8 );
    KMP_DEBUG_ASSERT( sizeof( kmp_intptr_t ) == sizeof( void * ) );

    __kmp_validate_locks();

    /* Initialize internal memory allocator */
    __kmp_init_allocator();

    /* Register the library startup via an environment variable
       and check to see whether another copy of the library is already
       registered. */

    __kmp_register_library_startup( );

    /* TODO reinitialization of library */
    if( TCR_4(__kmp_global.g.g_done) ) {
       KA_TRACE( 10, ("__kmp_do_serial_initialize: reinitialization of library\n" ) );
    }

    __kmp_global.g.g_abort = 0;
    TCW_SYNC_4(__kmp_global.g.g_done, FALSE);

    /* initialize the locks */
#if KMP_USE_ADAPTIVE_LOCKS
#if KMP_DEBUG_ADAPTIVE_LOCKS
    __kmp_init_speculative_stats();
#endif
#endif
    __kmp_init_lock( & __kmp_global_lock     );
    __kmp_init_queuing_lock( & __kmp_dispatch_lock );
    __kmp_init_lock( & __kmp_debug_lock      );
    __kmp_init_atomic_lock( & __kmp_atomic_lock     );
    __kmp_init_atomic_lock( & __kmp_atomic_lock_1i  );
    __kmp_init_atomic_lock( & __kmp_atomic_lock_2i  );
    __kmp_init_atomic_lock( & __kmp_atomic_lock_4i  );
    __kmp_init_atomic_lock( & __kmp_atomic_lock_4r  );
    __kmp_init_atomic_lock( & __kmp_atomic_lock_8i  );
    __kmp_init_atomic_lock( & __kmp_atomic_lock_8r  );
    __kmp_init_atomic_lock( & __kmp_atomic_lock_8c  );
    __kmp_init_atomic_lock( & __kmp_atomic_lock_10r );
    __kmp_init_atomic_lock( & __kmp_atomic_lock_16r );
    __kmp_init_atomic_lock( & __kmp_atomic_lock_16c );
    __kmp_init_atomic_lock( & __kmp_atomic_lock_20c );
    __kmp_init_atomic_lock( & __kmp_atomic_lock_32c );
    __kmp_init_bootstrap_lock( & __kmp_forkjoin_lock  );
    __kmp_init_bootstrap_lock( & __kmp_exit_lock      );
    __kmp_init_bootstrap_lock( & __kmp_monitor_lock   );
    __kmp_init_bootstrap_lock( & __kmp_tp_cached_lock );

    /* conduct initialization and initial setup of configuration */

    __kmp_runtime_initialize();

    // Some global variable initialization moved here from kmp_env_initialize()
#ifdef KMP_DEBUG
    kmp_diag = 0;
#endif
    __kmp_abort_delay = 0;

    // From __kmp_init_dflt_team_nth()
    /* assume the entire machine will be used */
    __kmp_dflt_team_nth_ub = __kmp_xproc;
    if( __kmp_dflt_team_nth_ub < KMP_MIN_NTH ) {
        __kmp_dflt_team_nth_ub = KMP_MIN_NTH;
    }
    if( __kmp_dflt_team_nth_ub > __kmp_sys_max_nth ) {
        __kmp_dflt_team_nth_ub = __kmp_sys_max_nth;
    }
    __kmp_max_nth = __kmp_sys_max_nth;

    // Three vars below moved here from __kmp_env_initialize() "KMP_BLOCKTIME" part
    __kmp_dflt_blocktime = KMP_DEFAULT_BLOCKTIME;
    __kmp_monitor_wakeups = KMP_WAKEUPS_FROM_BLOCKTIME( __kmp_dflt_blocktime, __kmp_monitor_wakeups );
    __kmp_bt_intervals = KMP_INTERVALS_FROM_BLOCKTIME( __kmp_dflt_blocktime, __kmp_monitor_wakeups );
    // From "KMP_LIBRARY" part of __kmp_env_initialize()
    __kmp_library = library_throughput;
    // From KMP_SCHEDULE initialization
    __kmp_static = kmp_sch_static_balanced;
    // AC: do not use analytical here, because it is non-monotonous
    //__kmp_guided = kmp_sch_guided_iterative_chunked;
    #if OMP_30_ENABLED
    //__kmp_auto = kmp_sch_guided_analytical_chunked; // AC: it is the default, no need to repeate assignment
    #endif // OMP_30_ENABLED
    // Barrier initialization. Moved here from __kmp_env_initialize() Barrier branch bit control and barrier method
    // control parts
    #if KMP_FAST_REDUCTION_BARRIER
        #define kmp_reduction_barrier_gather_bb ((int)1)
        #define kmp_reduction_barrier_release_bb ((int)1)
        #define kmp_reduction_barrier_gather_pat bp_hyper_bar
        #define kmp_reduction_barrier_release_pat bp_hyper_bar
    #endif // KMP_FAST_REDUCTION_BARRIER
    for ( i=bs_plain_barrier; i<bs_last_barrier; i++ ) {
        __kmp_barrier_gather_branch_bits [ i ] = __kmp_barrier_gather_bb_dflt;
        __kmp_barrier_release_branch_bits[ i ] = __kmp_barrier_release_bb_dflt;
        __kmp_barrier_gather_pattern [ i ] = __kmp_barrier_gather_pat_dflt;
        __kmp_barrier_release_pattern[ i ] = __kmp_barrier_release_pat_dflt;
        #if KMP_FAST_REDUCTION_BARRIER
        if( i == bs_reduction_barrier ) { // tested and confirmed on ALTIX only ( lin_64 ): hyper,1
            __kmp_barrier_gather_branch_bits [ i ] = kmp_reduction_barrier_gather_bb;
            __kmp_barrier_release_branch_bits[ i ] = kmp_reduction_barrier_release_bb;
            __kmp_barrier_gather_pattern [ i ] = kmp_reduction_barrier_gather_pat;
            __kmp_barrier_release_pattern[ i ] = kmp_reduction_barrier_release_pat;
        }
        #endif // KMP_FAST_REDUCTION_BARRIER
    }
    #if KMP_FAST_REDUCTION_BARRIER
        #undef kmp_reduction_barrier_release_pat
        #undef kmp_reduction_barrier_gather_pat
        #undef kmp_reduction_barrier_release_bb
        #undef kmp_reduction_barrier_gather_bb
    #endif // KMP_FAST_REDUCTION_BARRIER
    #if KMP_MIC
        // AC: plane=3,2, forkjoin=2,1 are optimal for 240 threads on KNC
        __kmp_barrier_gather_branch_bits [ 0 ] = 3;  // plane gather
        __kmp_barrier_release_branch_bits[ 1 ] = 1;  // forkjoin release
    #endif

    // From KMP_CHECKS initialization
#ifdef KMP_DEBUG
    __kmp_env_checks = TRUE;   /* development versions have the extra checks */
#else
    __kmp_env_checks = FALSE;  /* port versions do not have the extra checks */
#endif

    // From "KMP_FOREIGN_THREADS_THREADPRIVATE" initialization
    __kmp_foreign_tp = TRUE;

    __kmp_global.g.g_dynamic = FALSE;
    __kmp_global.g.g_dynamic_mode = dynamic_default;

    __kmp_env_initialize( NULL );
    // Print all messages in message catalog for testing purposes.
    #ifdef KMP_DEBUG
        char const * val = __kmp_env_get( "KMP_DUMP_CATALOG" );
        if ( __kmp_str_match_true( val ) ) {
            kmp_str_buf_t buffer;
            __kmp_str_buf_init( & buffer );
            __kmp_i18n_dump_catalog( & buffer );
            __kmp_printf( "%s", buffer.str );
            __kmp_str_buf_free( & buffer );
        }; // if
        __kmp_env_free( & val );
    #endif

    __kmp_threads_capacity = __kmp_initial_threads_capacity( __kmp_dflt_team_nth_ub );
    // Moved here from __kmp_env_initialize() "KMP_ALL_THREADPRIVATE" part
    __kmp_tp_capacity = __kmp_default_tp_capacity(__kmp_dflt_team_nth_ub, __kmp_max_nth, __kmp_allThreadsSpecified);


    // If the library is shut down properly, both pools must be NULL. Just in case, set them
    // to NULL -- some memory may leak, but subsequent code will work even if pools are not freed.
    KMP_DEBUG_ASSERT( __kmp_thread_pool == NULL );
    KMP_DEBUG_ASSERT( __kmp_thread_pool_insert_pt == NULL );
    KMP_DEBUG_ASSERT( __kmp_team_pool   == NULL );
    __kmp_thread_pool = NULL;
    __kmp_thread_pool_insert_pt = NULL;
    __kmp_team_pool   = NULL;

    /* Allocate all of the variable sized records */
    /* NOTE: __kmp_threads_capacity entries are allocated, but the arrays are expandable */
    /* Since allocation is cache-aligned, just add extra padding at the end */
    size = (sizeof(kmp_info_t*) + sizeof(kmp_root_t*))*__kmp_threads_capacity + CACHE_LINE;
    __kmp_threads = (kmp_info_t**) __kmp_allocate( size );
    __kmp_root    = (kmp_root_t**) ((char*)__kmp_threads + sizeof(kmp_info_t*) * __kmp_threads_capacity );

    /* init thread counts */
    KMP_DEBUG_ASSERT( __kmp_all_nth == 0 ); // Asserts fail if the library is reinitializing and
    KMP_DEBUG_ASSERT( __kmp_nth == 0 );     // something was wrong in termination.
    __kmp_all_nth = 0;
    __kmp_nth     = 0;

    /* setup the uber master thread and hierarchy */
    gtid = __kmp_register_root( TRUE );
    KA_TRACE( 10, ("__kmp_do_serial_initialize  T#%d\n", gtid ));
    KMP_ASSERT( KMP_UBER_GTID( gtid ) );
    KMP_ASSERT( KMP_INITIAL_GTID( gtid ) );

    KMP_MB();       /* Flush all pending memory write invalidates.  */

    __kmp_common_initialize();

    #if KMP_OS_UNIX
        /* invoke the child fork handler */
        __kmp_register_atfork();
    #endif

    #if ! defined GUIDEDLL_EXPORTS
        {
            /* Invoke the exit handler when the program finishes, only for static library.
               For dynamic library, we already have _fini and DllMain.
             */
            int rc = atexit( __kmp_internal_end_atexit );
            if ( rc != 0 ) {
                __kmp_msg( kmp_ms_fatal, KMP_MSG( FunctionError, "atexit()" ), KMP_ERR( rc ), __kmp_msg_null );
            }; // if
        }
    #endif

    #if KMP_HANDLE_SIGNALS
        #if KMP_OS_UNIX
            /* NOTE: make sure that this is called before the user installs
             *          their own signal handlers so that the user handlers
             *          are called first.  this way they can return false,
             *          not call our handler, avoid terminating the library,
             *          and continue execution where they left off. */
            __kmp_install_signals( FALSE );
        #endif /* KMP_OS_UNIX */
        #if KMP_OS_WINDOWS
            __kmp_install_signals( TRUE );
        #endif /* KMP_OS_WINDOWS */
    #endif

    /* we have finished the serial initialization */
    __kmp_init_counter ++;

    __kmp_init_serial = TRUE;

    if (__kmp_settings) {
        __kmp_env_print();
    }

#if OMP_40_ENABLED
    if (__kmp_display_env || __kmp_display_env_verbose) {
        __kmp_env_print_2();
    }
#endif // OMP_40_ENABLED

    KMP_MB();

    KA_TRACE( 10, ("__kmp_do_serial_initialize: exit\n" ) );
}

void
__kmp_serial_initialize( void )
{
    if ( __kmp_init_serial ) {
        return;
    }
    __kmp_acquire_bootstrap_lock( &__kmp_initz_lock );
    if ( __kmp_init_serial ) {
        __kmp_release_bootstrap_lock( &__kmp_initz_lock );
        return;
    }
    __kmp_do_serial_initialize();
    __kmp_release_bootstrap_lock( &__kmp_initz_lock );
}

static void
__kmp_do_middle_initialize( void )
{
    int i, j;
    int prev_dflt_team_nth;

    if( !__kmp_init_serial ) {
        __kmp_do_serial_initialize();
    }

    KA_TRACE( 10, ("__kmp_middle_initialize: enter\n" ) );

    //
    // Save the previous value for the __kmp_dflt_team_nth so that
    // we can avoid some reinitialization if it hasn't changed.
    //
    prev_dflt_team_nth = __kmp_dflt_team_nth;

#if KMP_AFFINITY_SUPPORTED
    //
    // __kmp_affinity_initialize() will try to set __kmp_ncores to the
    // number of cores on the machine.
    //
    __kmp_affinity_initialize();

    //
    // Run through the __kmp_threads array and set the affinity mask
    // for each root thread that is currently registered with the RTL.
    //
    for ( i = 0; i < __kmp_threads_capacity; i++ ) {
        if ( TCR_PTR( __kmp_threads[ i ] ) != NULL ) {
            __kmp_affinity_set_init_mask( i, TRUE );
        }
    }
#endif /* KMP_AFFINITY_SUPPORTED */

    KMP_ASSERT( __kmp_xproc > 0 );
    if ( __kmp_avail_proc == 0 ) {
        __kmp_avail_proc = __kmp_xproc;
    }

    // If there were empty places in num_threads list (OMP_NUM_THREADS=,,2,3), correct them now
    j = 0;
    while ( __kmp_nested_nth.used && ! __kmp_nested_nth.nth[ j ] ) {
        __kmp_nested_nth.nth[ j ] = __kmp_dflt_team_nth = __kmp_dflt_team_nth_ub = __kmp_avail_proc;
        j++;
    }

    if ( __kmp_dflt_team_nth == 0 ) {
#ifdef KMP_DFLT_NTH_CORES
        //
        // Default #threads = #cores
        //
        __kmp_dflt_team_nth = __kmp_ncores;
        KA_TRACE( 20, ("__kmp_middle_initialize: setting __kmp_dflt_team_nth = __kmp_ncores (%d)\n",
          __kmp_dflt_team_nth ) );
#else
        //
        // Default #threads = #available OS procs
        //
        __kmp_dflt_team_nth = __kmp_avail_proc;
        KA_TRACE( 20, ("__kmp_middle_initialize: setting __kmp_dflt_team_nth = __kmp_avail_proc(%d)\n",
          __kmp_dflt_team_nth ) );
#endif /* KMP_DFLT_NTH_CORES */
    }

    if ( __kmp_dflt_team_nth < KMP_MIN_NTH ) {
        __kmp_dflt_team_nth = KMP_MIN_NTH;
    }
    if( __kmp_dflt_team_nth > __kmp_sys_max_nth ) {
        __kmp_dflt_team_nth = __kmp_sys_max_nth;
    }

    //
    // There's no harm in continuing if the following check fails,
    // but it indicates an error in the previous logic.
    //
    KMP_DEBUG_ASSERT( __kmp_dflt_team_nth <= __kmp_dflt_team_nth_ub );

    if ( __kmp_dflt_team_nth != prev_dflt_team_nth ) {
        //
        // Run through the __kmp_threads array and set the num threads icv
        // for each root thread that is currently registered with the RTL
        // (which has not already explicitly set its nthreads-var with a
        // call to omp_set_num_threads()).
        //
        for ( i = 0; i < __kmp_threads_capacity; i++ ) {
            kmp_info_t *thread = __kmp_threads[ i ];
            if ( thread == NULL ) continue;
#if OMP_30_ENABLED
            if ( thread->th.th_current_task->td_icvs.nproc != 0 ) continue;
#else
            if ( thread->th.th_team->t.t_set_nproc[ thread->th.th_info.ds.ds_tid ]  != 0 ) continue;
#endif /* OMP_30_ENABLED */

            set__nproc_p( __kmp_threads[ i ], __kmp_dflt_team_nth );
        }
    }
    KA_TRACE( 20, ("__kmp_middle_initialize: final value for __kmp_dflt_team_nth = %d\n",
      __kmp_dflt_team_nth) );

#ifdef KMP_ADJUST_BLOCKTIME
    /* Adjust blocktime to zero if necessary */
    /* now that __kmp_avail_proc is set      */
    if ( !__kmp_env_blocktime && ( __kmp_avail_proc > 0 ) ) {
        KMP_DEBUG_ASSERT( __kmp_avail_proc > 0 );
        if ( __kmp_nth > __kmp_avail_proc ) {
            __kmp_zero_bt = TRUE;
        }
    }
#endif /* KMP_ADJUST_BLOCKTIME */

    /* we have finished middle initialization */
    TCW_SYNC_4(__kmp_init_middle, TRUE);

    KA_TRACE( 10, ("__kmp_do_middle_initialize: exit\n" ) );
}

void
__kmp_middle_initialize( void )
{
    if ( __kmp_init_middle ) {
        return;
    }
    __kmp_acquire_bootstrap_lock( &__kmp_initz_lock );
    if ( __kmp_init_middle ) {
        __kmp_release_bootstrap_lock( &__kmp_initz_lock );
        return;
    }
    __kmp_do_middle_initialize();
    __kmp_release_bootstrap_lock( &__kmp_initz_lock );
}

void
__kmp_parallel_initialize( void )
{
    int gtid = __kmp_entry_gtid();      // this might be a new root

    /* syncronize parallel initialization (for sibling) */
    if( TCR_4(__kmp_init_parallel) ) return;
    __kmp_acquire_bootstrap_lock( &__kmp_initz_lock );
    if( TCR_4(__kmp_init_parallel) ) { __kmp_release_bootstrap_lock( &__kmp_initz_lock ); return; }

    /* TODO reinitialization after we have already shut down */
    if( TCR_4(__kmp_global.g.g_done) ) {
        KA_TRACE( 10, ("__kmp_parallel_initialize: attempt to init while shutting down\n" ) );
        __kmp_infinite_loop();
    }

    /* jc: The lock __kmp_initz_lock is already held, so calling __kmp_serial_initialize
           would cause a deadlock.  So we call __kmp_do_serial_initialize directly.
    */
    if( !__kmp_init_middle ) {
        __kmp_do_middle_initialize();
    }

    /* begin initialization */
    KA_TRACE( 10, ("__kmp_parallel_initialize: enter\n" ) );
    KMP_ASSERT( KMP_UBER_GTID( gtid ) );

#if KMP_ARCH_X86 || KMP_ARCH_X86_64
    //
    // Save the FP control regs.
    // Worker threads will set theirs to these values at thread startup.
    //
    __kmp_store_x87_fpu_control_word( &__kmp_init_x87_fpu_control_word );
    __kmp_store_mxcsr( &__kmp_init_mxcsr );
    __kmp_init_mxcsr &= KMP_X86_MXCSR_MASK;
#endif /* KMP_ARCH_X86 || KMP_ARCH_X86_64 */

#if KMP_OS_UNIX
# if KMP_HANDLE_SIGNALS
    /*  must be after __kmp_serial_initialize  */
    __kmp_install_signals( TRUE );
# endif
#endif

    __kmp_suspend_initialize();

#  if defined(USE_LOAD_BALANCE)
    if ( __kmp_global.g.g_dynamic_mode == dynamic_default ) {
        __kmp_global.g.g_dynamic_mode = dynamic_load_balance;
    }
#else
    if ( __kmp_global.g.g_dynamic_mode == dynamic_default ) {
        __kmp_global.g.g_dynamic_mode = dynamic_thread_limit;
    }
#endif

    if ( __kmp_version ) {
        __kmp_print_version_2();
    }

    /* we have finished parallel initialization */
    TCW_SYNC_4(__kmp_init_parallel, TRUE);

    KMP_MB();
    KA_TRACE( 10, ("__kmp_parallel_initialize: exit\n" ) );

    __kmp_release_bootstrap_lock( &__kmp_initz_lock );
}


/* ------------------------------------------------------------------------ */

void
__kmp_run_before_invoked_task( int gtid, int tid, kmp_info_t *this_thr,
  kmp_team_t *team )
{
    kmp_disp_t *dispatch;

    KMP_MB();

    /* none of the threads have encountered any constructs, yet. */
    this_thr->th.th_local.this_construct = 0;
    this_thr->th.th_local.last_construct = 0;
#if KMP_CACHE_MANAGE
    KMP_CACHE_PREFETCH( &this_thr -> th.th_bar[ bs_forkjoin_barrier ].bb.b_arrived );
#endif /* KMP_CACHE_MANAGE */
    dispatch = (kmp_disp_t *)TCR_PTR(this_thr->th.th_dispatch);
    KMP_DEBUG_ASSERT( dispatch );
    KMP_DEBUG_ASSERT( team -> t.t_dispatch );
    //KMP_DEBUG_ASSERT( this_thr -> th.th_dispatch == &team -> t.t_dispatch[ this_thr->th.th_info.ds.ds_tid ] );

    dispatch -> th_disp_index = 0;    /* reset the dispatch buffer counter */

    if( __kmp_env_consistency_check )
        __kmp_push_parallel( gtid, team->t.t_ident );

    KMP_MB();       /* Flush all pending memory write invalidates.  */
}

void
__kmp_run_after_invoked_task( int gtid, int tid, kmp_info_t *this_thr,
  kmp_team_t *team )
{
    if( __kmp_env_consistency_check )
        __kmp_pop_parallel( gtid, team->t.t_ident );
}

int
__kmp_invoke_task_func( int gtid )
{
    int          rc;
    int          tid      = __kmp_tid_from_gtid( gtid );
    kmp_info_t  *this_thr = __kmp_threads[ gtid ];
    kmp_team_t  *team     = this_thr -> th.th_team;

    __kmp_run_before_invoked_task( gtid, tid, this_thr, team );
#if USE_ITT_BUILD
    if ( __itt_stack_caller_create_ptr ) {
        __kmp_itt_stack_callee_enter( (__itt_caller)team->t.t_stack_id ); // inform ittnotify about entering user's code
    }
#endif /* USE_ITT_BUILD */
    rc = __kmp_invoke_microtask( (microtask_t) TCR_SYNC_PTR(team->t.t_pkfn),
      gtid, tid, (int) team->t.t_argc, (void **) team->t.t_argv );

#if USE_ITT_BUILD
    if ( __itt_stack_caller_create_ptr ) {
        __kmp_itt_stack_callee_leave( (__itt_caller)team->t.t_stack_id ); // inform ittnotify about leaving user's code
    }
#endif /* USE_ITT_BUILD */
    __kmp_run_after_invoked_task( gtid, tid, this_thr, team );

    return rc;
}

#if OMP_40_ENABLED
void
__kmp_teams_master( microtask_t microtask, int gtid )
{
    // This routine is called by all master threads in teams construct
    kmp_info_t  *this_thr = __kmp_threads[ gtid ];
    kmp_team_t  *team = this_thr -> th.th_team;
    ident_t     *loc =  team->t.t_ident;

#if KMP_DEBUG
    int          tid = __kmp_tid_from_gtid( gtid );
    KA_TRACE( 20, ("__kmp_teams_master: T#%d, Tid %d, microtask %p\n",
                   gtid, tid, microtask) );
#endif

    // Launch league of teams now, but not let workers execute
    // (they hang on fork barrier until next parallel)
    this_thr->th.th_set_nproc = this_thr->th.th_set_nth_teams;
    __kmp_fork_call( loc, gtid, TRUE,
            team->t.t_argc,
            microtask,
            VOLATILE_CAST(launch_t) __kmp_invoke_task_func,
            NULL );
    __kmp_join_call( loc, gtid, 1 ); // AC: last parameter "1" eliminates join barrier which won't work because
                                     // worker threads are in a fork barrier waiting for more parallel regions
}

int
__kmp_invoke_teams_master( int gtid )
{
    #if KMP_DEBUG
    if ( !__kmp_threads[gtid]-> th.th_team->t.t_serialized )
        KMP_DEBUG_ASSERT( (void*)__kmp_threads[gtid]-> th.th_team->t.t_pkfn == (void*)__kmp_teams_master );
    #endif

    __kmp_teams_master( (microtask_t)__kmp_threads[gtid]->th.th_team_microtask, gtid );

    return 1;
}
#endif /* OMP_40_ENABLED */

/* this sets the requested number of threads for the next parallel region
 * encountered by this team */
/* since this should be enclosed in the forkjoin critical section it
 * should avoid race conditions with assymmetrical nested parallelism */

void
__kmp_push_num_threads( ident_t *id, int gtid, int num_threads )
{
    kmp_info_t *thr = __kmp_threads[gtid];

    if( num_threads > 0 )
        thr -> th.th_set_nproc = num_threads;
}

#if OMP_40_ENABLED

/* this sets the requested number of teams for the teams region and/or
 * the number of threads for the next parallel region encountered  */
void
__kmp_push_num_teams( ident_t *id, int gtid, int num_teams, int num_threads )
{
    kmp_info_t *thr = __kmp_threads[gtid];
    // The number of teams is the number of threads in the outer "parallel"
    if( num_teams > 0 ) {
        thr -> th.th_set_nproc = num_teams;
    } else {
        thr -> th.th_set_nproc = 1;  // AC: default number of teams is 1;
                                     // TODO: should it be __kmp_ncores ?
    }
    // The number of threads is for inner parallel regions
    if( num_threads > 0 ) {
        thr -> th.th_set_nth_teams = num_threads;
    } else {
        if( !TCR_4(__kmp_init_middle) )
            __kmp_middle_initialize();
        thr -> th.th_set_nth_teams = __kmp_avail_proc / thr -> th.th_set_nproc;
    }
}


//
// Set the proc_bind var to use in the following parallel region.
//
void
__kmp_push_proc_bind( ident_t *id, int gtid, kmp_proc_bind_t proc_bind )
{
    kmp_info_t *thr = __kmp_threads[gtid];
    thr -> th.th_set_proc_bind = proc_bind;
}

#endif /* OMP_40_ENABLED */

/* Launch the worker threads into the microtask. */

void
__kmp_internal_fork( ident_t *id, int gtid, kmp_team_t *team )
{
    kmp_info_t *this_thr = __kmp_threads[gtid];

#ifdef KMP_DEBUG
    int f;
#endif /* KMP_DEBUG */

    KMP_DEBUG_ASSERT( team );
    KMP_DEBUG_ASSERT( this_thr -> th.th_team  ==  team );
    KMP_ASSERT(       KMP_MASTER_GTID(gtid) );
    KMP_MB();       /* Flush all pending memory write invalidates.  */

    team -> t.t_construct = 0;          /* no single directives seen yet */
    team -> t.t_ordered.dt.t_value = 0; /* thread 0 enters the ordered section first */

    /* Reset the identifiers on the dispatch buffer */
    KMP_DEBUG_ASSERT( team -> t.t_disp_buffer );
    if ( team->t.t_max_nproc > 1 ) {
        int i;
        for (i = 0; i <  KMP_MAX_DISP_BUF; ++i)
            team -> t.t_disp_buffer[ i ].buffer_index = i;
    } else {
        team -> t.t_disp_buffer[ 0 ].buffer_index = 0;
    }

    KMP_MB();       /* Flush all pending memory write invalidates.  */
    KMP_ASSERT( this_thr -> th.th_team  ==  team );

#ifdef KMP_DEBUG
    for( f=0 ; f<team->t.t_nproc ; f++ ) {
        KMP_DEBUG_ASSERT( team->t.t_threads[f] &&
                          team->t.t_threads[f]->th.th_team_nproc == team->t.t_nproc );
    }
#endif /* KMP_DEBUG */

    /* release the worker threads so they may begin working */
    __kmp_fork_barrier( gtid, 0 );
}


void
__kmp_internal_join( ident_t *id, int gtid, kmp_team_t *team )
{
    kmp_info_t *this_thr = __kmp_threads[gtid];

    KMP_DEBUG_ASSERT( team );
    KMP_DEBUG_ASSERT( this_thr -> th.th_team  ==  team );
    KMP_ASSERT(       KMP_MASTER_GTID(gtid) );
    KMP_MB();       /* Flush all pending memory write invalidates.  */

    /* Join barrier after fork */

#ifdef KMP_DEBUG
    if (__kmp_threads[gtid] && __kmp_threads[gtid]->th.th_team_nproc != team->t.t_nproc ) {
        __kmp_printf("GTID: %d, __kmp_threads[%d]=%p\n",gtid, gtid, __kmp_threads[gtid]);
        __kmp_printf("__kmp_threads[%d]->th.th_team_nproc=%d, TEAM: %p, team->t.t_nproc=%d\n",
                     gtid, __kmp_threads[gtid]->th.th_team_nproc, team, team->t.t_nproc);
        __kmp_print_structure();
    }
    KMP_DEBUG_ASSERT( __kmp_threads[gtid] &&
                     __kmp_threads[gtid]->th.th_team_nproc == team->t.t_nproc );
#endif /* KMP_DEBUG */

    __kmp_join_barrier( gtid );  /* wait for everyone */

    KMP_MB();       /* Flush all pending memory write invalidates.  */
    KMP_ASSERT( this_thr -> th.th_team  ==  team );
}


/* ------------------------------------------------------------------------ */
/* ------------------------------------------------------------------------ */

#ifdef USE_LOAD_BALANCE

//
// Return the worker threads actively spinning in the hot team, if we
// are at the outermost level of parallelism.  Otherwise, return 0.
//
static int
__kmp_active_hot_team_nproc( kmp_root_t *root )
{
    int i;
    int retval;
    kmp_team_t *hot_team;

    if ( root->r.r_active ) {
        return 0;
    }
    hot_team = root->r.r_hot_team;
    if ( __kmp_dflt_blocktime == KMP_MAX_BLOCKTIME ) {
        return hot_team->t.t_nproc - 1;  // Don't count master thread
    }

    //
    // Skip the master thread - it is accounted for elsewhere.
    //
    retval = 0;
    for ( i = 1; i < hot_team->t.t_nproc; i++ ) {
        if ( hot_team->t.t_threads[i]->th.th_active ) {
            retval++;
        }
    }
    return retval;
}

//
// Perform an automatic adjustment to the number of
// threads used by the next parallel region.
//
static int
__kmp_load_balance_nproc( kmp_root_t *root, int set_nproc )
{
    int retval;
    int pool_active;
    int hot_team_active;
    int team_curr_active;
    int system_active;

    KB_TRACE( 20, ("__kmp_load_balance_nproc: called root:%p set_nproc:%d\n",
                root, set_nproc ) );
    KMP_DEBUG_ASSERT( root );
    #if OMP_30_ENABLED
    KMP_DEBUG_ASSERT( root->r.r_root_team->t.t_threads[0]->th.th_current_task->td_icvs.dynamic == TRUE );
    #else
    KMP_DEBUG_ASSERT( root->r.r_root_team->t.t_set_dynamic[0] == TRUE );
    #endif
    KMP_DEBUG_ASSERT( set_nproc > 1 );

    if ( set_nproc == 1) {
        KB_TRACE( 20, ("__kmp_load_balance_nproc: serial execution.\n" ) );
        return 1;
    }

    //
    // Threads that are active in the thread pool, active in the hot team
    // for this particular root (if we are at the outer par level), and
    // the currently executing thread (to become the master) are available
    // to add to the new team, but are currently contributing to the system
    // load, and must be accounted for.
    //
    pool_active = TCR_4(__kmp_thread_pool_active_nth);
    hot_team_active = __kmp_active_hot_team_nproc( root );
    team_curr_active = pool_active + hot_team_active + 1;

    //
    // Check the system load.
    //
    system_active = __kmp_get_load_balance( __kmp_avail_proc + team_curr_active );
    KB_TRACE( 30, ("__kmp_load_balance_nproc: system active = %d pool active = %d hot team active = %d\n",
      system_active, pool_active, hot_team_active ) );

    if ( system_active < 0 ) {
        //
        // There was an error reading the necessary info from /proc,
        // so use the thread limit algorithm instead.  Once we set
        // __kmp_global.g.g_dynamic_mode = dynamic_thread_limit,
        // we shouldn't wind up getting back here.
        //
        __kmp_global.g.g_dynamic_mode = dynamic_thread_limit;
        KMP_WARNING( CantLoadBalUsing, "KMP_DYNAMIC_MODE=thread limit" );

        //
        // Make this call behave like the thread limit algorithm.
        //
        retval = __kmp_avail_proc - __kmp_nth + (root->r.r_active ? 1
          : root->r.r_hot_team->t.t_nproc);
        if ( retval > set_nproc ) {
            retval = set_nproc;
        }
        if ( retval < KMP_MIN_NTH ) {
            retval = KMP_MIN_NTH;
        }

        KB_TRACE( 20, ("__kmp_load_balance_nproc: thread limit exit. retval:%d\n", retval ) );
        return retval;
    }

    //
    // There is a slight delay in the load balance algorithm in detecting
    // new running procs.  The real system load at this instant should be
    // at least as large as the #active omp thread that are available to
    // add to the team.
    //
    if ( system_active < team_curr_active ) {
        system_active = team_curr_active;
    }
    retval = __kmp_avail_proc - system_active + team_curr_active;
    if ( retval > set_nproc ) {
        retval = set_nproc;
    }
    if ( retval < KMP_MIN_NTH ) {
        retval = KMP_MIN_NTH;
    }

    KB_TRACE( 20, ("__kmp_load_balance_nproc: exit. retval:%d\n", retval ) );
    return retval;
} // __kmp_load_balance_nproc()

#endif /* USE_LOAD_BALANCE */


/* ------------------------------------------------------------------------ */
/* ------------------------------------------------------------------------ */

/* NOTE: this is called with the __kmp_init_lock held */
void
__kmp_cleanup( void )
{
    int f;

    KA_TRACE( 10, ("__kmp_cleanup: enter\n" ) );

    if (TCR_4(__kmp_init_parallel)) {
#if KMP_HANDLE_SIGNALS
        __kmp_remove_signals();
#endif
        TCW_4(__kmp_init_parallel, FALSE);
    }

    if (TCR_4(__kmp_init_middle)) {
#if KMP_AFFINITY_SUPPORTED
        __kmp_affinity_uninitialize();
#endif /* KMP_AFFINITY_SUPPORTED */
        TCW_4(__kmp_init_middle, FALSE);
    }

    KA_TRACE( 10, ("__kmp_cleanup: go serial cleanup\n" ) );

    if (__kmp_init_serial) {

        __kmp_runtime_destroy();

        __kmp_init_serial = FALSE;
    }

    for ( f = 0; f < __kmp_threads_capacity; f++ ) {
        if ( __kmp_root[ f ] != NULL ) {
            __kmp_free( __kmp_root[ f ] );
            __kmp_root[ f ] = NULL;
        }
    }
    __kmp_free( __kmp_threads );
    // __kmp_threads and __kmp_root were allocated at once, as single block, so there is no need in
    // freeing __kmp_root.
    __kmp_threads = NULL;
    __kmp_root    = NULL;
    __kmp_threads_capacity = 0;

    __kmp_cleanup_user_locks();

    #if KMP_AFFINITY_SUPPORTED
        KMP_INTERNAL_FREE( (void *) __kmp_cpuinfo_file );
        __kmp_cpuinfo_file = NULL;
    #endif /* KMP_AFFINITY_SUPPORTED */

   #if KMP_USE_ADAPTIVE_LOCKS
   #if KMP_DEBUG_ADAPTIVE_LOCKS
       __kmp_print_speculative_stats();
   #endif
   #endif
    KMP_INTERNAL_FREE( __kmp_nested_nth.nth );
    __kmp_nested_nth.nth = NULL;
    __kmp_nested_nth.size = 0;
    __kmp_nested_nth.used = 0;

    __kmp_i18n_catclose();

    KA_TRACE( 10, ("__kmp_cleanup: exit\n" ) );
}

/* ------------------------------------------------------------------------ */
/* ------------------------------------------------------------------------ */

int
__kmp_ignore_mppbeg( void )
{
    char *env;

    if ((env = getenv( "KMP_IGNORE_MPPBEG" )) != NULL) {
        if (__kmp_str_match_false( env ))
            return FALSE;
    }
    // By default __kmpc_begin() is no-op.
    return TRUE;
}

int
__kmp_ignore_mppend( void )
{
    char *env;

    if ((env = getenv( "KMP_IGNORE_MPPEND" )) != NULL) {
        if (__kmp_str_match_false( env ))
            return FALSE;
    }
    // By default __kmpc_end() is no-op.
    return TRUE;
}

void
__kmp_internal_begin( void )
{
    int gtid;
    kmp_root_t *root;

    /* this is a very important step as it will register new sibling threads
     * and assign these new uber threads a new gtid */
    gtid = __kmp_entry_gtid();
    root = __kmp_threads[ gtid ] -> th.th_root;
    KMP_ASSERT( KMP_UBER_GTID( gtid ));

    if( root->r.r_begin ) return;
    __kmp_acquire_lock( &root->r.r_begin_lock, gtid );
    if( root->r.r_begin ) {
        __kmp_release_lock( & root->r.r_begin_lock, gtid );
        return;
    }

    root -> r.r_begin = TRUE;

    __kmp_release_lock( & root->r.r_begin_lock, gtid );
}


/* ------------------------------------------------------------------------ */
/* ------------------------------------------------------------------------ */

void
__kmp_user_set_library (enum library_type arg)
{
    int gtid;
    kmp_root_t *root;
    kmp_info_t *thread;

    /* first, make sure we are initialized so we can get our gtid */

    gtid = __kmp_entry_gtid();
    thread = __kmp_threads[ gtid ];

    root = thread -> th.th_root;

    KA_TRACE( 20, ("__kmp_user_set_library: enter T#%d, arg: %d, %d\n", gtid, arg, library_serial ));
    if (root->r.r_in_parallel) { /* Must be called in serial section of top-level thread */
        KMP_WARNING( SetLibraryIncorrectCall );
        return;
    }

    switch ( arg ) {
    case library_serial :
        thread -> th.th_set_nproc = 0;
        set__nproc_p( thread, 1 );
        break;
    case library_turnaround :
        thread -> th.th_set_nproc = 0;
        set__nproc_p( thread, __kmp_dflt_team_nth ? __kmp_dflt_team_nth : __kmp_dflt_team_nth_ub );
        break;
    case library_throughput :
        thread -> th.th_set_nproc = 0;
        set__nproc_p( thread, __kmp_dflt_team_nth ? __kmp_dflt_team_nth : __kmp_dflt_team_nth_ub );
        break;
    default:
        KMP_FATAL( UnknownLibraryType, arg );
    }

    __kmp_aux_set_library ( arg );
}

void
__kmp_aux_set_stacksize( size_t arg )
{
    if (! __kmp_init_serial)
        __kmp_serial_initialize();

#if KMP_OS_DARWIN
    if (arg & (0x1000 - 1)) {
        arg &= ~(0x1000 - 1);
        if(arg + 0x1000) /* check for overflow if we round up */
            arg += 0x1000;
    }
#endif
    __kmp_acquire_bootstrap_lock( &__kmp_initz_lock );

    /* only change the default stacksize before the first parallel region */
    if (! TCR_4(__kmp_init_parallel)) {
        size_t value = arg;       /* argument is in bytes */

        if (value < __kmp_sys_min_stksize )
            value = __kmp_sys_min_stksize ;
        else if (value > KMP_MAX_STKSIZE)
            value = KMP_MAX_STKSIZE;

        __kmp_stksize = value;

        __kmp_env_stksize = TRUE;    /* was KMP_STACKSIZE specified? */
    }

    __kmp_release_bootstrap_lock( &__kmp_initz_lock );
}

/* set the behaviour of the runtime library */
/* TODO this can cause some odd behaviour with sibling parallelism... */
void
__kmp_aux_set_library (enum library_type arg)
{
    __kmp_library = arg;

    switch ( __kmp_library ) {
    case library_serial :
        {
            KMP_INFORM( LibraryIsSerial );
            (void) __kmp_change_library( TRUE );
        }
        break;
    case library_turnaround :
        (void) __kmp_change_library( TRUE );
        break;
    case library_throughput :
        (void) __kmp_change_library( FALSE );
        break;
    default:
        KMP_FATAL( UnknownLibraryType, arg );
    }
}

/* ------------------------------------------------------------------------ */
/* ------------------------------------------------------------------------ */

void
__kmp_aux_set_blocktime (int arg, kmp_info_t *thread, int tid)
{
    int blocktime = arg;        /* argument is in milliseconds */
    int bt_intervals;
    int bt_set;

    __kmp_save_internal_controls( thread );

    /* Normalize and set blocktime for the teams */
    if (blocktime < KMP_MIN_BLOCKTIME)
        blocktime = KMP_MIN_BLOCKTIME;
    else if (blocktime > KMP_MAX_BLOCKTIME)
        blocktime = KMP_MAX_BLOCKTIME;

    set__blocktime_team( thread -> th.th_team, tid, blocktime );
    set__blocktime_team( thread -> th.th_serial_team, 0, blocktime );

    /* Calculate and set blocktime intervals for the teams */
    bt_intervals = KMP_INTERVALS_FROM_BLOCKTIME(blocktime, __kmp_monitor_wakeups);

    set__bt_intervals_team( thread -> th.th_team, tid, bt_intervals );
    set__bt_intervals_team( thread -> th.th_serial_team, 0, bt_intervals );

    /* Set whether blocktime has been set to "TRUE" */
    bt_set = TRUE;

    set__bt_set_team( thread -> th.th_team, tid, bt_set );
    set__bt_set_team( thread -> th.th_serial_team, 0, bt_set );
    KF_TRACE(10, ( "kmp_set_blocktime: T#%d(%d:%d), blocktime=%d, bt_intervals=%d, monitor_updates=%d\n",
                  __kmp_gtid_from_tid(tid, thread->th.th_team),
                  thread->th.th_team->t.t_id, tid, blocktime, bt_intervals, __kmp_monitor_wakeups ) );
}

void
__kmp_aux_set_defaults(
    char const * str,
    int          len
) {
    if ( ! __kmp_init_serial ) {
        __kmp_serial_initialize();
    };
    __kmp_env_initialize( str );

    if (__kmp_settings
#if OMP_40_ENABLED
        || __kmp_display_env || __kmp_display_env_verbose
#endif // OMP_40_ENABLED
        ) {
        __kmp_env_print();
    }
} // __kmp_aux_set_defaults

/* ------------------------------------------------------------------------ */

/*
 * internal fast reduction routines
 */

PACKED_REDUCTION_METHOD_T
__kmp_determine_reduction_method( ident_t *loc, kmp_int32 global_tid,
        kmp_int32 num_vars, size_t reduce_size, void *reduce_data, void (*reduce_func)(void *lhs_data, void *rhs_data),
        kmp_critical_name *lck )
{

    // Default reduction method: critical construct ( lck != NULL, like in current PAROPT )
    // If ( reduce_data!=NULL && reduce_func!=NULL ): the tree-reduction method can be selected by RTL
    // If loc->flags contains KMP_IDENT_ATOMIC_REDUCE, the atomic reduce method can be selected by RTL
    // Finally, it's up to OpenMP RTL to make a decision on which method to select among generated by PAROPT.

    PACKED_REDUCTION_METHOD_T retval;

    int team_size;

    KMP_DEBUG_ASSERT( loc );    // it would be nice to test ( loc != 0 )
    KMP_DEBUG_ASSERT( lck );    // it would be nice to test ( lck != 0 )

    #define FAST_REDUCTION_ATOMIC_METHOD_GENERATED ( ( loc->flags & ( KMP_IDENT_ATOMIC_REDUCE ) ) == ( KMP_IDENT_ATOMIC_REDUCE ) )
    #define FAST_REDUCTION_TREE_METHOD_GENERATED   ( ( reduce_data ) && ( reduce_func ) )

    retval = critical_reduce_block;

    team_size = __kmp_get_team_num_threads( global_tid ); // another choice of getting a team size ( with 1 dynamic deference ) is slower

    if( team_size == 1 ) {

        retval = empty_reduce_block;

    } else {

        int atomic_available = FAST_REDUCTION_ATOMIC_METHOD_GENERATED;
        int tree_available   = FAST_REDUCTION_TREE_METHOD_GENERATED;

        #if KMP_ARCH_X86_64 || KMP_ARCH_PPC64

            #if KMP_OS_LINUX || KMP_OS_FREEBSD || KMP_OS_WINDOWS || KMP_OS_DARWIN
                #if KMP_MIC
                    #define REDUCTION_TEAMSIZE_CUTOFF 8
                #else // KMP_MIC
                    #define REDUCTION_TEAMSIZE_CUTOFF 4
                #endif // KMP_MIC
                if( tree_available ) {
                    if( team_size <= REDUCTION_TEAMSIZE_CUTOFF ) {
                        if ( atomic_available ) {
                            retval = atomic_reduce_block;
                        }
                    } else {
                        retval = TREE_REDUCE_BLOCK_WITH_REDUCTION_BARRIER;
                    }
                } else if ( atomic_available ) {
                    retval = atomic_reduce_block;
                }
            #else
                #error "Unknown or unsupported OS"
            #endif // KMP_OS_LINUX || KMP_OS_FREEBSD || KMP_OS_WINDOWS || KMP_OS_DARWIN

        #elif KMP_ARCH_X86 || KMP_ARCH_ARM

            #if KMP_AFFINITY_SUPPORTED

                // basic tuning

                if( atomic_available ) {
                    if( num_vars <= 2 ) { // && ( team_size <= 8 ) due to false-sharing ???
                        retval = atomic_reduce_block;
                    }
                } // otherwise: use critical section

            #elif KMP_OS_DARWIN

                if( atomic_available && ( num_vars <= 3 ) ) {
                        retval = atomic_reduce_block;
                } else if( tree_available ) {
                    if( ( reduce_size > ( 9 * sizeof( kmp_real64 ) ) ) && ( reduce_size < ( 2000 * sizeof( kmp_real64 ) ) ) ) {
                        retval = TREE_REDUCE_BLOCK_WITH_PLAIN_BARRIER;
                    }
                } // otherwise: use critical section

            #else
                #error "Unknown or unsupported OS"
            #endif

        #else
            #error "Unknown or unsupported architecture"
        #endif

    }

    // KMP_FORCE_REDUCTION

    if( __kmp_force_reduction_method != reduction_method_not_defined ) {

        PACKED_REDUCTION_METHOD_T forced_retval;

        int atomic_available, tree_available;

        switch( ( forced_retval = __kmp_force_reduction_method ) )
        {
            case critical_reduce_block:
                KMP_ASSERT( lck );              // lck should be != 0
                if( team_size <= 1 ) {
                    forced_retval = empty_reduce_block;
                }
                break;

            case atomic_reduce_block:
                atomic_available = FAST_REDUCTION_ATOMIC_METHOD_GENERATED;
                KMP_ASSERT( atomic_available ); // atomic_available should be != 0
                break;

            case tree_reduce_block:
                tree_available = FAST_REDUCTION_TREE_METHOD_GENERATED;
                KMP_ASSERT( tree_available );   // tree_available should be != 0
                #if KMP_FAST_REDUCTION_BARRIER
                forced_retval = TREE_REDUCE_BLOCK_WITH_REDUCTION_BARRIER;
                #endif
                break;

            default:
                KMP_ASSERT( 0 ); // "unsupported method specified"
        }

        retval = forced_retval;
    }

    KA_TRACE(10, ( "reduction method selected=%08x\n", retval ) );

    #undef FAST_REDUCTION_TREE_METHOD_GENERATED
    #undef FAST_REDUCTION_ATOMIC_METHOD_GENERATED

    return ( retval );
}

// this function is for testing set/get/determine reduce method
kmp_int32
__kmp_get_reduce_method( void ) {
    return ( ( __kmp_entry_thread() -> th.th_local.packed_reduction_method ) >> 8 );
}

/* ------------------------------------------------------------------------ */
