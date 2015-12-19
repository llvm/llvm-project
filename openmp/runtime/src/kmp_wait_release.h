/*
 * kmp_wait_release.h -- Wait/Release implementation
 */


//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


#ifndef KMP_WAIT_RELEASE_H
#define KMP_WAIT_RELEASE_H

#include "kmp.h"
#include "kmp_itt.h"

/*!
@defgroup WAIT_RELEASE Wait/Release operations

The definitions and functions here implement the lowest level thread
synchronizations of suspending a thread and awaking it. They are used
to build higher level operations such as barriers and fork/join.
*/

/*!
@ingroup WAIT_RELEASE
@{
*/

/*! 
 * The flag_type describes the storage used for the flag.
 */
enum flag_type {
    flag32,        /**< 32 bit flags */
    flag64,        /**< 64 bit flags */
    flag_oncore    /**< special 64-bit flag for on-core barrier (hierarchical) */
};

/*!
 * Base class for wait/release volatile flag
 */
template <typename P>
class kmp_flag {
    volatile P * loc;  /**< Pointer to the flag storage that is modified by another thread */
    flag_type t;       /**< "Type" of the flag in loc */
 public:
    typedef P flag_t;
    kmp_flag(volatile P *p, flag_type ft) : loc(p), t(ft) {}
    /*!
     * @result the pointer to the actual flag
     */
    volatile P * get() { return loc; }
    /*!
     * @param new_loc in   set loc to point at new_loc
     */
    void set(volatile P *new_loc) { loc = new_loc; }
    /*!
     * @result the flag_type
     */
    flag_type get_type() { return t; }
    // Derived classes must provide the following:
    /*
    kmp_info_t * get_waiter(kmp_uint32 i);
    kmp_uint32 get_num_waiters();
    bool done_check();
    bool done_check_val(P old_loc);
    bool notdone_check();
    P internal_release();
    void suspend(int th_gtid);
    void resume(int th_gtid);
    P set_sleeping();
    P unset_sleeping();
    bool is_sleeping();
    bool is_any_sleeping();
    bool is_sleeping_val(P old_loc);
    int execute_tasks(kmp_info_t *this_thr, kmp_int32 gtid, int final_spin, int *thread_finished
                      USE_ITT_BUILD_ARG(void * itt_sync_obj), kmp_int32 is_constrained);
    */
};

/* Spin wait loop that first does pause, then yield, then sleep. A thread that calls __kmp_wait_*
   must make certain that another thread calls __kmp_release to wake it back up to prevent deadlocks!  */
template <class C>
static inline void __kmp_wait_template(kmp_info_t *this_thr, C *flag, int final_spin
                                       USE_ITT_BUILD_ARG(void * itt_sync_obj) )
{
    // NOTE: We may not belong to a team at this point.
    volatile typename C::flag_t *spin = flag->get();
    kmp_uint32 spins;
    kmp_uint32 hibernate;
    int th_gtid;
    int tasks_completed = FALSE;

    KMP_FSYNC_SPIN_INIT(spin, NULL);
    if (flag->done_check()) {
        KMP_FSYNC_SPIN_ACQUIRED(spin);
        return;
    }
    th_gtid = this_thr->th.th_info.ds.ds_gtid;
    KA_TRACE(20, ("__kmp_wait_sleep: T#%d waiting for flag(%p)\n", th_gtid, flag));

#if OMPT_SUPPORT && OMPT_BLAME
    ompt_state_t ompt_state = this_thr->th.ompt_thread_info.state;
    if (ompt_enabled &&
        ompt_state != ompt_state_undefined) {
        if (ompt_state == ompt_state_idle) {
            if (ompt_callbacks.ompt_callback(ompt_event_idle_begin)) {
                ompt_callbacks.ompt_callback(ompt_event_idle_begin)(th_gtid + 1);
            }
        } else if (ompt_callbacks.ompt_callback(ompt_event_wait_barrier_begin)) {
            KMP_DEBUG_ASSERT(ompt_state == ompt_state_wait_barrier ||
                             ompt_state == ompt_state_wait_barrier_implicit ||
                             ompt_state == ompt_state_wait_barrier_explicit);

            ompt_lw_taskteam_t* team = this_thr->th.th_team->t.ompt_serialized_team_info;
            ompt_parallel_id_t pId;
            ompt_task_id_t tId;
            if (team){
                pId = team->ompt_team_info.parallel_id;
                tId = team->ompt_task_info.task_id;
            } else {
                pId = this_thr->th.th_team->t.ompt_team_info.parallel_id;
                tId = this_thr->th.th_current_task->ompt_task_info.task_id;
            }
            ompt_callbacks.ompt_callback(ompt_event_wait_barrier_begin)(pId, tId);
        }
    }
#endif

    // Setup for waiting
    KMP_INIT_YIELD(spins);

    if (__kmp_dflt_blocktime != KMP_MAX_BLOCKTIME) {
        // The worker threads cannot rely on the team struct existing at this point.
        // Use the bt values cached in the thread struct instead.
#ifdef KMP_ADJUST_BLOCKTIME
        if (__kmp_zero_bt && !this_thr->th.th_team_bt_set)
            // Force immediate suspend if not set by user and more threads than available procs
            hibernate = 0;
        else
            hibernate = this_thr->th.th_team_bt_intervals;
#else
        hibernate = this_thr->th.th_team_bt_intervals;
#endif /* KMP_ADJUST_BLOCKTIME */

        /* If the blocktime is nonzero, we want to make sure that we spin wait for the entirety
           of the specified #intervals, plus up to one interval more.  This increment make
           certain that this thread doesn't go to sleep too soon.  */
        if (hibernate != 0)
            hibernate++;

        // Add in the current time value.
        hibernate += TCR_4(__kmp_global.g.g_time.dt.t_value);
        KF_TRACE(20, ("__kmp_wait_sleep: T#%d now=%d, hibernate=%d, intervals=%d\n",
                      th_gtid, __kmp_global.g.g_time.dt.t_value, hibernate,
                      hibernate - __kmp_global.g.g_time.dt.t_value));
    }

    KMP_MB();

    // Main wait spin loop
    while (flag->notdone_check()) {
        int in_pool;

        /* If the task team is NULL, it means one of things:
           1) A newly-created thread is first being released by __kmp_fork_barrier(), and
              its task team has not been set up yet.
           2) All tasks have been executed to completion, this thread has decremented the task
              team's ref ct and possibly deallocated it, and should no longer reference it.
           3) Tasking is off for this region.  This could be because we are in a serialized region
              (perhaps the outer one), or else tasking was manually disabled (KMP_TASKING=0).  */
        kmp_task_team_t * task_team = NULL;
        if (__kmp_tasking_mode != tskm_immediate_exec) {
            task_team = this_thr->th.th_task_team;
            if (task_team != NULL) {
                if (TCR_SYNC_4(task_team->tt.tt_active)) {
                    if (KMP_TASKING_ENABLED(task_team))
                        flag->execute_tasks(this_thr, th_gtid, final_spin, &tasks_completed
                                            USE_ITT_BUILD_ARG(itt_sync_obj), 0);
                }
                else {
                    KMP_DEBUG_ASSERT(!KMP_MASTER_TID(this_thr->th.th_info.ds.ds_tid));
                    this_thr->th.th_task_team = NULL;
                }
            } // if
        } // if

        KMP_FSYNC_SPIN_PREPARE(spin);
        if (TCR_4(__kmp_global.g.g_done)) {
            if (__kmp_global.g.g_abort)
                __kmp_abort_thread();
            break;
        }

        // If we are oversubscribed, or have waited a bit (and KMP_LIBRARY=throughput), then yield
        KMP_YIELD(TCR_4(__kmp_nth) > __kmp_avail_proc);
        // TODO: Should it be number of cores instead of thread contexts? Like:
        // KMP_YIELD(TCR_4(__kmp_nth) > __kmp_ncores);
        // Need performance improvement data to make the change...
        KMP_YIELD_SPIN(spins);

        // Check if this thread was transferred from a team
        // to the thread pool (or vice-versa) while spinning.
        in_pool = !!TCR_4(this_thr->th.th_in_pool);
        if (in_pool != !!this_thr->th.th_active_in_pool) {
            if (in_pool) { // Recently transferred from team to pool
                KMP_TEST_THEN_INC32((kmp_int32 *)&__kmp_thread_pool_active_nth);
                this_thr->th.th_active_in_pool = TRUE;
                /* Here, we cannot assert that:
                   KMP_DEBUG_ASSERT(TCR_4(__kmp_thread_pool_active_nth) <= __kmp_thread_pool_nth);
                   __kmp_thread_pool_nth is inc/dec'd by the master thread while the fork/join
                   lock is held, whereas __kmp_thread_pool_active_nth is inc/dec'd asynchronously
                   by the workers.  The two can get out of sync for brief periods of time.  */
            }
            else { // Recently transferred from pool to team
                KMP_TEST_THEN_DEC32((kmp_int32 *) &__kmp_thread_pool_active_nth);
                KMP_DEBUG_ASSERT(TCR_4(__kmp_thread_pool_active_nth) >= 0);
                this_thr->th.th_active_in_pool = FALSE;
            }
        }

        // Don't suspend if KMP_BLOCKTIME is set to "infinite"
        if (__kmp_dflt_blocktime == KMP_MAX_BLOCKTIME)
            continue;

        // Don't suspend if there is a likelihood of new tasks being spawned.
        if ((task_team != NULL) && TCR_4(task_team->tt.tt_found_tasks))
            continue;

        // If we have waited a bit more, fall asleep
        if (TCR_4(__kmp_global.g.g_time.dt.t_value) < hibernate)
            continue;

        KF_TRACE(50, ("__kmp_wait_sleep: T#%d suspend time reached\n", th_gtid));

        flag->suspend(th_gtid);

        if (TCR_4(__kmp_global.g.g_done)) {
            if (__kmp_global.g.g_abort)
                __kmp_abort_thread();
            break;
        }
        // TODO: If thread is done with work and times out, disband/free
    }

#if OMPT_SUPPORT && OMPT_BLAME
    if (ompt_enabled &&
        ompt_state != ompt_state_undefined) {
        if (ompt_state == ompt_state_idle) {
            if (ompt_callbacks.ompt_callback(ompt_event_idle_end)) {
                ompt_callbacks.ompt_callback(ompt_event_idle_end)(th_gtid + 1);
            }
        } else if (ompt_callbacks.ompt_callback(ompt_event_wait_barrier_end)) {
            KMP_DEBUG_ASSERT(ompt_state == ompt_state_wait_barrier ||
                             ompt_state == ompt_state_wait_barrier_implicit ||
                             ompt_state == ompt_state_wait_barrier_explicit);

            ompt_lw_taskteam_t* team = this_thr->th.th_team->t.ompt_serialized_team_info;
            ompt_parallel_id_t pId;
            ompt_task_id_t tId;
            if (team){
                pId = team->ompt_team_info.parallel_id;
                tId = team->ompt_task_info.task_id;
            } else {
                pId = this_thr->th.th_team->t.ompt_team_info.parallel_id;
                tId = this_thr->th.th_current_task->ompt_task_info.task_id;
            }
            ompt_callbacks.ompt_callback(ompt_event_wait_barrier_end)(pId, tId);
        }
    }
#endif

    KMP_FSYNC_SPIN_ACQUIRED(spin);
}

/* Release any threads specified as waiting on the flag by releasing the flag and resume the waiting thread
   if indicated by the sleep bit(s). A thread that calls __kmp_wait_template must call this function to wake
   up the potentially sleeping thread and prevent deadlocks!  */
template <class C>
static inline void __kmp_release_template(C *flag)
{
#ifdef KMP_DEBUG
    int gtid = TCR_4(__kmp_init_gtid) ? __kmp_get_gtid() : -1;
#endif
    KF_TRACE(20, ("__kmp_release: T#%d releasing flag(%x)\n", gtid, flag->get()));
    KMP_DEBUG_ASSERT(flag->get());
    KMP_FSYNC_RELEASING(flag->get());

    flag->internal_release();

    KF_TRACE(100, ("__kmp_release: T#%d set new spin=%d\n", gtid, flag->get(), *(flag->get())));

    if (__kmp_dflt_blocktime != KMP_MAX_BLOCKTIME) {
        // Only need to check sleep stuff if infinite block time not set
        if (flag->is_any_sleeping()) { // Are *any* of the threads that wait on this flag sleeping?
            for (unsigned int i=0; i<flag->get_num_waiters(); ++i) {
                kmp_info_t * waiter = flag->get_waiter(i); // if a sleeping waiter exists at i, sets current_waiter to i inside the flag
                if (waiter) {
                    int wait_gtid = waiter->th.th_info.ds.ds_gtid;
                    // Wake up thread if needed
                    KF_TRACE(50, ("__kmp_release: T#%d waking up thread T#%d since sleep flag(%p) set\n",
                                  gtid, wait_gtid, flag->get()));
                    flag->resume(wait_gtid); // unsets flag's current_waiter when done
                }
            }
        }
    }
}

template <typename FlagType>
struct flag_traits {};

template <>
struct flag_traits<kmp_uint32> {
    typedef kmp_uint32 flag_t;
    static const flag_type t = flag32;
    static inline flag_t tcr(flag_t f) { return TCR_4(f); }
    static inline flag_t test_then_add4(volatile flag_t *f) { return KMP_TEST_THEN_ADD4_32((volatile kmp_int32 *)f); }
    static inline flag_t test_then_or(volatile flag_t *f, flag_t v) { return KMP_TEST_THEN_OR32((volatile kmp_int32 *)f, v); }
    static inline flag_t test_then_and(volatile flag_t *f, flag_t v) { return KMP_TEST_THEN_AND32((volatile kmp_int32 *)f, v); }
};

template <>
struct flag_traits<kmp_uint64> {
    typedef kmp_uint64 flag_t;
    static const flag_type t = flag64;
    static inline flag_t tcr(flag_t f) { return TCR_8(f); }
    static inline flag_t test_then_add4(volatile flag_t *f) { return KMP_TEST_THEN_ADD4_64((volatile kmp_int64 *)f); }
    static inline flag_t test_then_or(volatile flag_t *f, flag_t v) { return KMP_TEST_THEN_OR64((volatile kmp_int64 *)f, v); }
    static inline flag_t test_then_and(volatile flag_t *f, flag_t v) { return KMP_TEST_THEN_AND64((volatile kmp_int64 *)f, v); }
};

template <typename FlagType>
class kmp_basic_flag : public kmp_flag<FlagType> {
    typedef flag_traits<FlagType> traits_type;
    FlagType checker;  /**< Value to compare flag to to check if flag has been released. */
    kmp_info_t * waiting_threads[1];  /**< Array of threads sleeping on this thread. */
    kmp_uint32 num_waiting_threads;       /**< Number of threads sleeping on this thread. */
 public:
    kmp_basic_flag(volatile FlagType *p) : kmp_flag<FlagType>(p, traits_type::t), num_waiting_threads(0) {}
    kmp_basic_flag(volatile FlagType *p, kmp_info_t *thr) : kmp_flag<FlagType>(p, traits_type::t), num_waiting_threads(1) {
        waiting_threads[0] = thr; 
    }
    kmp_basic_flag(volatile FlagType *p, FlagType c) : kmp_flag<FlagType>(p, traits_type::t), checker(c), num_waiting_threads(0) {}
    /*!
     * param i in   index into waiting_threads
     * @result the thread that is waiting at index i
     */
    kmp_info_t * get_waiter(kmp_uint32 i) { 
        KMP_DEBUG_ASSERT(i<num_waiting_threads);
        return waiting_threads[i]; 
    }
    /*!
     * @result num_waiting_threads
     */
    kmp_uint32 get_num_waiters() { return num_waiting_threads; }
    /*!
     * @param thr in   the thread which is now waiting
     *
     * Insert a waiting thread at index 0.
     */
    void set_waiter(kmp_info_t *thr) { 
        waiting_threads[0] = thr; 
        num_waiting_threads = 1;
    }
    /*!
     * @result true if the flag object has been released.
     */
    bool done_check() { return traits_type::tcr(*(this->get())) == checker; }
    /*!
     * @param old_loc in   old value of flag
     * @result true if the flag's old value indicates it was released.
     */
    bool done_check_val(FlagType old_loc) { return old_loc == checker; }
    /*!
     * @result true if the flag object is not yet released.
     * Used in __kmp_wait_template like:
     * @code
     * while (flag.notdone_check()) { pause(); }
     * @endcode
     */
    bool notdone_check() { return traits_type::tcr(*(this->get())) != checker; }
    /*!
     * @result Actual flag value before release was applied.
     * Trigger all waiting threads to run by modifying flag to release state.
     */
    void internal_release() {
        (void) traits_type::test_then_add4((volatile FlagType *)this->get());
    }
    /*!
     * @result Actual flag value before sleep bit(s) set.
     * Notes that there is at least one thread sleeping on the flag by setting sleep bit(s).
     */
    FlagType set_sleeping() { 
        return traits_type::test_then_or((volatile FlagType *)this->get(), KMP_BARRIER_SLEEP_STATE);
    }
    /*!
     * @result Actual flag value before sleep bit(s) cleared.
     * Notes that there are no longer threads sleeping on the flag by clearing sleep bit(s).
     */
    FlagType unset_sleeping() { 
        return traits_type::test_then_and((volatile FlagType *)this->get(), ~KMP_BARRIER_SLEEP_STATE);
    }
    /*! 
     * @param old_loc in   old value of flag
     * Test whether there are threads sleeping on the flag's old value in old_loc.
     */
    bool is_sleeping_val(FlagType old_loc) { return old_loc & KMP_BARRIER_SLEEP_STATE; }
    /*! 
     * Test whether there are threads sleeping on the flag.
     */
    bool is_sleeping() { return is_sleeping_val(*(this->get())); }
    bool is_any_sleeping() { return is_sleeping_val(*(this->get())); }
    kmp_uint8 *get_stolen() { return NULL; }
    enum barrier_type get_bt() { return bs_last_barrier; }
};

class kmp_flag_32 : public kmp_basic_flag<kmp_uint32> {
 public:
    kmp_flag_32(volatile kmp_uint32 *p) : kmp_basic_flag<kmp_uint32>(p) {}
    kmp_flag_32(volatile kmp_uint32 *p, kmp_info_t *thr) : kmp_basic_flag<kmp_uint32>(p, thr) {}
    kmp_flag_32(volatile kmp_uint32 *p, kmp_uint32 c) : kmp_basic_flag<kmp_uint32>(p, c) {}
    void suspend(int th_gtid) { __kmp_suspend_32(th_gtid, this); }
    void resume(int th_gtid) { __kmp_resume_32(th_gtid, this); }
    int execute_tasks(kmp_info_t *this_thr, kmp_int32 gtid, int final_spin, int *thread_finished
                      USE_ITT_BUILD_ARG(void * itt_sync_obj), kmp_int32 is_constrained) {
        return __kmp_execute_tasks_32(this_thr, gtid, this, final_spin, thread_finished
                                      USE_ITT_BUILD_ARG(itt_sync_obj), is_constrained);
    }
    void wait(kmp_info_t *this_thr, int final_spin
              USE_ITT_BUILD_ARG(void * itt_sync_obj)) {
        __kmp_wait_template(this_thr, this, final_spin
                            USE_ITT_BUILD_ARG(itt_sync_obj));
    }
    void release() { __kmp_release_template(this); }
    flag_type get_ptr_type() { return flag32; }
};

class kmp_flag_64 : public kmp_basic_flag<kmp_uint64> {
 public:
    kmp_flag_64(volatile kmp_uint64 *p) : kmp_basic_flag<kmp_uint64>(p) {}
    kmp_flag_64(volatile kmp_uint64 *p, kmp_info_t *thr) : kmp_basic_flag<kmp_uint64>(p, thr) {}
    kmp_flag_64(volatile kmp_uint64 *p, kmp_uint64 c) : kmp_basic_flag<kmp_uint64>(p, c) {}
    void suspend(int th_gtid) { __kmp_suspend_64(th_gtid, this); }
    void resume(int th_gtid) { __kmp_resume_64(th_gtid, this); }
    int execute_tasks(kmp_info_t *this_thr, kmp_int32 gtid, int final_spin, int *thread_finished
                      USE_ITT_BUILD_ARG(void * itt_sync_obj), kmp_int32 is_constrained) {
        return __kmp_execute_tasks_64(this_thr, gtid, this, final_spin, thread_finished
                                      USE_ITT_BUILD_ARG(itt_sync_obj), is_constrained);
    }
    void wait(kmp_info_t *this_thr, int final_spin
              USE_ITT_BUILD_ARG(void * itt_sync_obj)) {
        __kmp_wait_template(this_thr, this, final_spin
                            USE_ITT_BUILD_ARG(itt_sync_obj));
    }
    void release() { __kmp_release_template(this); }
    flag_type get_ptr_type() { return flag64; }
};

// Hierarchical 64-bit on-core barrier instantiation
class kmp_flag_oncore : public kmp_flag<kmp_uint64> {
    kmp_uint64 checker;
    kmp_info_t * waiting_threads[1];
    kmp_uint32 num_waiting_threads;
    kmp_uint32 offset;      /**< Portion of flag that is of interest for an operation. */
    bool flag_switch;       /**< Indicates a switch in flag location. */
    enum barrier_type bt;   /**< Barrier type. */
    kmp_info_t * this_thr;  /**< Thread that may be redirected to different flag location. */
#if USE_ITT_BUILD
    void *itt_sync_obj;     /**< ITT object that must be passed to new flag location. */
#endif
    unsigned char& byteref(volatile kmp_uint64* loc, size_t offset) { return ((unsigned char *)loc)[offset]; }
public:
    kmp_flag_oncore(volatile kmp_uint64 *p)
        : kmp_flag<kmp_uint64>(p, flag_oncore), num_waiting_threads(0), flag_switch(false) {}
    kmp_flag_oncore(volatile kmp_uint64 *p, kmp_uint32 idx)
        : kmp_flag<kmp_uint64>(p, flag_oncore), num_waiting_threads(0), offset(idx), flag_switch(false) {}
    kmp_flag_oncore(volatile kmp_uint64 *p, kmp_uint64 c, kmp_uint32 idx, enum barrier_type bar_t,
                    kmp_info_t * thr
#if USE_ITT_BUILD
                    , void *itt
#endif
                    )
        : kmp_flag<kmp_uint64>(p, flag_oncore), checker(c), num_waiting_threads(0), offset(idx),
          flag_switch(false), bt(bar_t), this_thr(thr)
#if USE_ITT_BUILD
        , itt_sync_obj(itt)
#endif
        {}
    kmp_info_t * get_waiter(kmp_uint32 i) {
        KMP_DEBUG_ASSERT(i<num_waiting_threads);
        return waiting_threads[i];
    }
    kmp_uint32 get_num_waiters() { return num_waiting_threads; }
    void set_waiter(kmp_info_t *thr) {
        waiting_threads[0] = thr;
        num_waiting_threads = 1;
    }
    bool done_check_val(kmp_uint64 old_loc) { return byteref(&old_loc,offset) == checker; }
    bool done_check() { return done_check_val(*get()); }
    bool notdone_check() {
        // Calculate flag_switch
        if (this_thr->th.th_bar[bt].bb.wait_flag == KMP_BARRIER_SWITCH_TO_OWN_FLAG)
            flag_switch = true;
        if (byteref(get(),offset) != 1 && !flag_switch)
            return true;
        else if (flag_switch) {
            this_thr->th.th_bar[bt].bb.wait_flag = KMP_BARRIER_SWITCHING;
            kmp_flag_64 flag(&this_thr->th.th_bar[bt].bb.b_go, (kmp_uint64)KMP_BARRIER_STATE_BUMP);
            __kmp_wait_64(this_thr, &flag, TRUE
#if USE_ITT_BUILD
                          , itt_sync_obj
#endif
                          );
        }
        return false;
    }
    void internal_release() {
        if (__kmp_dflt_blocktime == KMP_MAX_BLOCKTIME) {
            byteref(get(),offset) = 1;
        }
        else {
            kmp_uint64 mask=0;
            byteref(&mask,offset) = 1;
            (void) KMP_TEST_THEN_OR64((volatile kmp_int64 *)get(), mask);
        }
    }
    kmp_uint64 set_sleeping() {
        return KMP_TEST_THEN_OR64((kmp_int64 volatile *)get(), KMP_BARRIER_SLEEP_STATE);
    }
    kmp_uint64 unset_sleeping() {
        return KMP_TEST_THEN_AND64((kmp_int64 volatile *)get(), ~KMP_BARRIER_SLEEP_STATE);
    }
    bool is_sleeping_val(kmp_uint64 old_loc) { return old_loc & KMP_BARRIER_SLEEP_STATE; }
    bool is_sleeping() { return is_sleeping_val(*get()); }
    bool is_any_sleeping() { return is_sleeping_val(*get()); }
    void wait(kmp_info_t *this_thr, int final_spin) {
        __kmp_wait_template<kmp_flag_oncore>(this_thr, this, final_spin
                            USE_ITT_BUILD_ARG(itt_sync_obj));
    }
    void release() { __kmp_release_template(this); }
    void suspend(int th_gtid) { __kmp_suspend_oncore(th_gtid, this); }
    void resume(int th_gtid) { __kmp_resume_oncore(th_gtid, this); }
    int execute_tasks(kmp_info_t *this_thr, kmp_int32 gtid, int final_spin, int *thread_finished
                      USE_ITT_BUILD_ARG(void * itt_sync_obj), kmp_int32 is_constrained) {
        return __kmp_execute_tasks_oncore(this_thr, gtid, this, final_spin, thread_finished
                                          USE_ITT_BUILD_ARG(itt_sync_obj), is_constrained);
    }
    kmp_uint8 *get_stolen() { return NULL; }
    enum barrier_type get_bt() { return bt; }
    flag_type get_ptr_type() { return flag_oncore; }
};


/*!
@}
*/

#endif // KMP_WAIT_RELEASE_H
