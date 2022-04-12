/*
 * kmp_wait_release.h -- Wait/Release implementation
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef KMP_WAIT_RELEASE_H
#define KMP_WAIT_RELEASE_H

#include "kmp.h"
#include "kmp_itt.h"
#include "kmp_stats.h"
#if OMPT_SUPPORT
#include "ompt-specific.h"
#endif

/*!
@defgroup WAIT_RELEASE Wait/Release operations

The definitions and functions here implement the lowest level thread
synchronizations of suspending a thread and awaking it. They are used to build
higher level operations such as barriers and fork/join.
*/

/*!
@ingroup WAIT_RELEASE
@{
*/

struct flag_properties {
  unsigned int type : 16;
  unsigned int reserved : 16;
};

template <enum flag_type FlagType> struct flag_traits {};

template <> struct flag_traits<flag32> {
  typedef kmp_uint32 flag_t;
  static const flag_type t = flag32;
  static inline flag_t tcr(flag_t f) { return TCR_4(f); }
  static inline flag_t test_then_add4(volatile flag_t *f) {
    return KMP_TEST_THEN_ADD4_32(RCAST(volatile kmp_int32 *, f));
  }
  static inline flag_t test_then_or(volatile flag_t *f, flag_t v) {
    return KMP_TEST_THEN_OR32(f, v);
  }
  static inline flag_t test_then_and(volatile flag_t *f, flag_t v) {
    return KMP_TEST_THEN_AND32(f, v);
  }
};

template <> struct flag_traits<atomic_flag64> {
  typedef kmp_uint64 flag_t;
  static const flag_type t = atomic_flag64;
  static inline flag_t tcr(flag_t f) { return TCR_8(f); }
  static inline flag_t test_then_add4(volatile flag_t *f) {
    return KMP_TEST_THEN_ADD4_64(RCAST(volatile kmp_int64 *, f));
  }
  static inline flag_t test_then_or(volatile flag_t *f, flag_t v) {
    return KMP_TEST_THEN_OR64(f, v);
  }
  static inline flag_t test_then_and(volatile flag_t *f, flag_t v) {
    return KMP_TEST_THEN_AND64(f, v);
  }
};

template <> struct flag_traits<flag64> {
  typedef kmp_uint64 flag_t;
  static const flag_type t = flag64;
  static inline flag_t tcr(flag_t f) { return TCR_8(f); }
  static inline flag_t test_then_add4(volatile flag_t *f) {
    return KMP_TEST_THEN_ADD4_64(RCAST(volatile kmp_int64 *, f));
  }
  static inline flag_t test_then_or(volatile flag_t *f, flag_t v) {
    return KMP_TEST_THEN_OR64(f, v);
  }
  static inline flag_t test_then_and(volatile flag_t *f, flag_t v) {
    return KMP_TEST_THEN_AND64(f, v);
  }
};

template <> struct flag_traits<flag_oncore> {
  typedef kmp_uint64 flag_t;
  static const flag_type t = flag_oncore;
  static inline flag_t tcr(flag_t f) { return TCR_8(f); }
  static inline flag_t test_then_add4(volatile flag_t *f) {
    return KMP_TEST_THEN_ADD4_64(RCAST(volatile kmp_int64 *, f));
  }
  static inline flag_t test_then_or(volatile flag_t *f, flag_t v) {
    return KMP_TEST_THEN_OR64(f, v);
  }
  static inline flag_t test_then_and(volatile flag_t *f, flag_t v) {
    return KMP_TEST_THEN_AND64(f, v);
  }
};

/*! Base class for all flags */
template <flag_type FlagType> class kmp_flag {
protected:
  flag_properties t; /**< "Type" of the flag in loc */
  kmp_info_t *waiting_threads[1]; /**< Threads sleeping on this thread. */
  kmp_uint32 num_waiting_threads; /**< Num threads sleeping on this thread. */
  std::atomic<bool> *sleepLoc;

public:
  typedef flag_traits<FlagType> traits_type;
  kmp_flag() : t({FlagType, 0U}), num_waiting_threads(0), sleepLoc(nullptr) {}
  kmp_flag(int nwaiters)
      : t({FlagType, 0U}), num_waiting_threads(nwaiters), sleepLoc(nullptr) {}
  kmp_flag(std::atomic<bool> *sloc)
      : t({FlagType, 0U}), num_waiting_threads(0), sleepLoc(sloc) {}
  /*! @result the flag_type */
  flag_type get_type() { return (flag_type)(t.type); }

  /*! param i in   index into waiting_threads
   *  @result the thread that is waiting at index i */
  kmp_info_t *get_waiter(kmp_uint32 i) {
    KMP_DEBUG_ASSERT(i < num_waiting_threads);
    return waiting_threads[i];
  }
  /*! @result num_waiting_threads */
  kmp_uint32 get_num_waiters() { return num_waiting_threads; }
  /*! @param thr in   the thread which is now waiting
   *  Insert a waiting thread at index 0. */
  void set_waiter(kmp_info_t *thr) {
    waiting_threads[0] = thr;
    num_waiting_threads = 1;
  }
  enum barrier_type get_bt() { return bs_last_barrier; }
};

/*! Base class for wait/release volatile flag */
template <typename PtrType, flag_type FlagType, bool Sleepable>
class kmp_flag_native : public kmp_flag<FlagType> {
protected:
  volatile PtrType *loc;
  PtrType checker; /**< When flag==checker, it has been released. */
  typedef flag_traits<FlagType> traits_type;

public:
  typedef PtrType flag_t;
  kmp_flag_native(volatile PtrType *p) : kmp_flag<FlagType>(), loc(p) {}
  kmp_flag_native(volatile PtrType *p, kmp_info_t *thr)
      : kmp_flag<FlagType>(1), loc(p) {
    this->waiting_threads[0] = thr;
  }
  kmp_flag_native(volatile PtrType *p, PtrType c)
      : kmp_flag<FlagType>(), loc(p), checker(c) {}
  kmp_flag_native(volatile PtrType *p, PtrType c, std::atomic<bool> *sloc)
      : kmp_flag<FlagType>(sloc), loc(p), checker(c) {}
  virtual ~kmp_flag_native() {}
  void *operator new(size_t size) { return __kmp_allocate(size); }
  void operator delete(void *p) { __kmp_free(p); }
  volatile PtrType *get() { return loc; }
  void *get_void_p() { return RCAST(void *, CCAST(PtrType *, loc)); }
  void set(volatile PtrType *new_loc) { loc = new_loc; }
  PtrType load() { return *loc; }
  void store(PtrType val) { *loc = val; }
  /*! @result true if the flag object has been released. */
  virtual bool done_check() {
    if (Sleepable && !(this->sleepLoc))
      return (traits_type::tcr(*(this->get())) & ~KMP_BARRIER_SLEEP_STATE) ==
             checker;
    else
      return traits_type::tcr(*(this->get())) == checker;
  }
  /*! @param old_loc in   old value of flag
   *  @result true if the flag's old value indicates it was released. */
  virtual bool done_check_val(PtrType old_loc) { return old_loc == checker; }
  /*! @result true if the flag object is not yet released.
   * Used in __kmp_wait_template like:
   * @code
   * while (flag.notdone_check()) { pause(); }
   * @endcode */
  virtual bool notdone_check() {
    return traits_type::tcr(*(this->get())) != checker;
  }
  /*! @result Actual flag value before release was applied.
   * Trigger all waiting threads to run by modifying flag to release state. */
  void internal_release() {
    (void)traits_type::test_then_add4((volatile PtrType *)this->get());
  }
  /*! @result Actual flag value before sleep bit(s) set.
   * Notes that there is at least one thread sleeping on the flag by setting
   * sleep bit(s). */
  PtrType set_sleeping() {
    if (this->sleepLoc) {
      this->sleepLoc->store(true);
      return *(this->get());
    }
    return traits_type::test_then_or((volatile PtrType *)this->get(),
                                     KMP_BARRIER_SLEEP_STATE);
  }
  /*! @result Actual flag value before sleep bit(s) cleared.
   * Notes that there are no longer threads sleeping on the flag by clearing
   * sleep bit(s). */
  void unset_sleeping() {
    if (this->sleepLoc) {
      this->sleepLoc->store(false);
      return;
    }
    traits_type::test_then_and((volatile PtrType *)this->get(),
                               ~KMP_BARRIER_SLEEP_STATE);
  }
  /*! @param old_loc in   old value of flag
   * Test if there are threads sleeping on the flag's old value in old_loc. */
  bool is_sleeping_val(PtrType old_loc) {
    if (this->sleepLoc)
      return this->sleepLoc->load();
    return old_loc & KMP_BARRIER_SLEEP_STATE;
  }
  /*! Test whether there are threads sleeping on the flag. */
  bool is_sleeping() {
    if (this->sleepLoc)
      return this->sleepLoc->load();
    return is_sleeping_val(*(this->get()));
  }
  bool is_any_sleeping() {
    if (this->sleepLoc)
      return this->sleepLoc->load();
    return is_sleeping_val(*(this->get()));
  }
  kmp_uint8 *get_stolen() { return NULL; }
};

/*! Base class for wait/release atomic flag */
template <typename PtrType, flag_type FlagType, bool Sleepable>
class kmp_flag_atomic : public kmp_flag<FlagType> {
protected:
  std::atomic<PtrType> *loc; /**< Pointer to flag location to wait on */
  PtrType checker; /**< Flag == checker means it has been released. */
public:
  typedef flag_traits<FlagType> traits_type;
  typedef PtrType flag_t;
  kmp_flag_atomic(std::atomic<PtrType> *p) : kmp_flag<FlagType>(), loc(p) {}
  kmp_flag_atomic(std::atomic<PtrType> *p, kmp_info_t *thr)
      : kmp_flag<FlagType>(1), loc(p) {
    this->waiting_threads[0] = thr;
  }
  kmp_flag_atomic(std::atomic<PtrType> *p, PtrType c)
      : kmp_flag<FlagType>(), loc(p), checker(c) {}
  kmp_flag_atomic(std::atomic<PtrType> *p, PtrType c, std::atomic<bool> *sloc)
      : kmp_flag<FlagType>(sloc), loc(p), checker(c) {}
  /*! @result the pointer to the actual flag */
  std::atomic<PtrType> *get() { return loc; }
  /*! @result void* pointer to the actual flag */
  void *get_void_p() { return RCAST(void *, loc); }
  /*! @param new_loc in   set loc to point at new_loc */
  void set(std::atomic<PtrType> *new_loc) { loc = new_loc; }
  /*! @result flag value */
  PtrType load() { return loc->load(std::memory_order_acquire); }
  /*! @param val the new flag value to be stored */
  void store(PtrType val) { loc->store(val, std::memory_order_release); }
  /*! @result true if the flag object has been released. */
  bool done_check() {
    if (Sleepable && !(this->sleepLoc))
      return (this->load() & ~KMP_BARRIER_SLEEP_STATE) == checker;
    else
      return this->load() == checker;
  }
  /*! @param old_loc in   old value of flag
   * @result true if the flag's old value indicates it was released. */
  bool done_check_val(PtrType old_loc) { return old_loc == checker; }
  /*! @result true if the flag object is not yet released.
   * Used in __kmp_wait_template like:
   * @code
   * while (flag.notdone_check()) { pause(); }
   * @endcode */
  bool notdone_check() { return this->load() != checker; }
  /*! @result Actual flag value before release was applied.
   * Trigger all waiting threads to run by modifying flag to release state. */
  void internal_release() { KMP_ATOMIC_ADD(this->get(), 4); }
  /*! @result Actual flag value before sleep bit(s) set.
   * Notes that there is at least one thread sleeping on the flag by setting
   * sleep bit(s). */
  PtrType set_sleeping() {
    if (this->sleepLoc) {
      this->sleepLoc->store(true);
      return *(this->get());
    }
    return KMP_ATOMIC_OR(this->get(), KMP_BARRIER_SLEEP_STATE);
  }
  /*! @result Actual flag value before sleep bit(s) cleared.
   * Notes that there are no longer threads sleeping on the flag by clearing
   * sleep bit(s). */
  void unset_sleeping() {
    if (this->sleepLoc) {
      this->sleepLoc->store(false);
      return;
    }
    KMP_ATOMIC_AND(this->get(), ~KMP_BARRIER_SLEEP_STATE);
  }
  /*! @param old_loc in   old value of flag
   * Test whether there are threads sleeping on flag's old value in old_loc. */
  bool is_sleeping_val(PtrType old_loc) {
    if (this->sleepLoc)
      return this->sleepLoc->load();
    return old_loc & KMP_BARRIER_SLEEP_STATE;
  }
  /*! Test whether there are threads sleeping on the flag. */
  bool is_sleeping() {
    if (this->sleepLoc)
      return this->sleepLoc->load();
    return is_sleeping_val(this->load());
  }
  bool is_any_sleeping() {
    if (this->sleepLoc)
      return this->sleepLoc->load();
    return is_sleeping_val(this->load());
  }
  kmp_uint8 *get_stolen() { return NULL; }
};

#if OMPT_SUPPORT
OMPT_NOINLINE
static void __ompt_implicit_task_end(kmp_info_t *this_thr,
                                     ompt_state_t ompt_state,
                                     ompt_data_t *tId) {
  int ds_tid = this_thr->th.th_info.ds.ds_tid;
  if (ompt_state == ompt_state_wait_barrier_implicit) {
    this_thr->th.ompt_thread_info.state = ompt_state_overhead;
#if OMPT_OPTIONAL
    void *codeptr = NULL;
    if (ompt_enabled.ompt_callback_sync_region_wait) {
      ompt_callbacks.ompt_callback(ompt_callback_sync_region_wait)(
          ompt_sync_region_barrier_implicit, ompt_scope_end, NULL, tId,
          codeptr);
    }
    if (ompt_enabled.ompt_callback_sync_region) {
      ompt_callbacks.ompt_callback(ompt_callback_sync_region)(
          ompt_sync_region_barrier_implicit, ompt_scope_end, NULL, tId,
          codeptr);
    }
#endif
    if (!KMP_MASTER_TID(ds_tid)) {
      if (ompt_enabled.ompt_callback_implicit_task) {
        int flags = this_thr->th.ompt_thread_info.parallel_flags;
        flags = (flags & ompt_parallel_league) ? ompt_task_initial
                                               : ompt_task_implicit;
        ompt_callbacks.ompt_callback(ompt_callback_implicit_task)(
            ompt_scope_end, NULL, tId, 0, ds_tid, flags);
      }
      // return to idle state
      this_thr->th.ompt_thread_info.state = ompt_state_idle;
    } else {
      this_thr->th.ompt_thread_info.state = ompt_state_overhead;
    }
  }
}
#endif

/* Spin wait loop that first does pause/yield, then sleep. A thread that calls
   __kmp_wait_*  must make certain that another thread calls __kmp_release
   to wake it back up to prevent deadlocks!

   NOTE: We may not belong to a team at this point.  */
template <class C, bool final_spin, bool Cancellable = false,
          bool Sleepable = true>
static inline bool
__kmp_wait_template(kmp_info_t *this_thr,
                    C *flag USE_ITT_BUILD_ARG(void *itt_sync_obj)) {
#if USE_ITT_BUILD && USE_ITT_NOTIFY
  volatile void *spin = flag->get();
#endif
  kmp_uint32 spins;
  int th_gtid;
  int tasks_completed = FALSE;
#if !KMP_USE_MONITOR
  kmp_uint64 poll_count;
  kmp_uint64 hibernate_goal;
#else
  kmp_uint32 hibernate;
#endif
  kmp_uint64 time;

  KMP_FSYNC_SPIN_INIT(spin, NULL);
  if (flag->done_check()) {
    KMP_FSYNC_SPIN_ACQUIRED(CCAST(void *, spin));
    return false;
  }
  th_gtid = this_thr->th.th_info.ds.ds_gtid;
  if (Cancellable) {
    kmp_team_t *team = this_thr->th.th_team;
    if (team && team->t.t_cancel_request == cancel_parallel)
      return true;
  }
#if KMP_OS_UNIX
  if (final_spin)
    KMP_ATOMIC_ST_REL(&this_thr->th.th_blocking, true);
#endif
  KA_TRACE(20,
           ("__kmp_wait_sleep: T#%d waiting for flag(%p)\n", th_gtid, flag));
#if KMP_STATS_ENABLED
  stats_state_e thread_state = KMP_GET_THREAD_STATE();
#endif

/* OMPT Behavior:
THIS function is called from
  __kmp_barrier (2 times)  (implicit or explicit barrier in parallel regions)
            these have join / fork behavior

       In these cases, we don't change the state or trigger events in THIS
function.
       Events are triggered in the calling code (__kmp_barrier):

                state := ompt_state_overhead
            barrier-begin
            barrier-wait-begin
                state := ompt_state_wait_barrier
          call join-barrier-implementation (finally arrive here)
          {}
          call fork-barrier-implementation (finally arrive here)
          {}
                state := ompt_state_overhead
            barrier-wait-end
            barrier-end
                state := ompt_state_work_parallel


  __kmp_fork_barrier  (after thread creation, before executing implicit task)
          call fork-barrier-implementation (finally arrive here)
          {} // worker arrive here with state = ompt_state_idle


  __kmp_join_barrier  (implicit barrier at end of parallel region)
                state := ompt_state_barrier_implicit
            barrier-begin
            barrier-wait-begin
          call join-barrier-implementation (finally arrive here
final_spin=FALSE)
          {
          }
  __kmp_fork_barrier  (implicit barrier at end of parallel region)
          call fork-barrier-implementation (finally arrive here final_spin=TRUE)

       Worker after task-team is finished:
            barrier-wait-end
            barrier-end
            implicit-task-end
            idle-begin
                state := ompt_state_idle

       Before leaving, if state = ompt_state_idle
            idle-end
                state := ompt_state_overhead
*/
#if OMPT_SUPPORT
  ompt_state_t ompt_entry_state;
  ompt_data_t *tId;
  if (ompt_enabled.enabled) {
    ompt_entry_state = this_thr->th.ompt_thread_info.state;
    if (!final_spin || ompt_entry_state != ompt_state_wait_barrier_implicit ||
        KMP_MASTER_TID(this_thr->th.th_info.ds.ds_tid)) {
      ompt_lw_taskteam_t *team = NULL;
      if (this_thr->th.th_team)
        team = this_thr->th.th_team->t.ompt_serialized_team_info;
      if (team) {
        tId = &(team->ompt_task_info.task_data);
      } else {
        tId = OMPT_CUR_TASK_DATA(this_thr);
      }
    } else {
      tId = &(this_thr->th.ompt_thread_info.task_data);
    }
    if (final_spin && (__kmp_tasking_mode == tskm_immediate_exec ||
                       this_thr->th.th_task_team == NULL)) {
      // implicit task is done. Either no taskqueue, or task-team finished
      __ompt_implicit_task_end(this_thr, ompt_entry_state, tId);
    }
  }
#endif

  KMP_INIT_YIELD(spins); // Setup for waiting
  KMP_INIT_BACKOFF(time);

  if (__kmp_dflt_blocktime != KMP_MAX_BLOCKTIME ||
      __kmp_pause_status == kmp_soft_paused) {
#if KMP_USE_MONITOR
// The worker threads cannot rely on the team struct existing at this point.
// Use the bt values cached in the thread struct instead.
#ifdef KMP_ADJUST_BLOCKTIME
    if (__kmp_pause_status == kmp_soft_paused ||
        (__kmp_zero_bt && !this_thr->th.th_team_bt_set))
      // Force immediate suspend if not set by user and more threads than
      // available procs
      hibernate = 0;
    else
      hibernate = this_thr->th.th_team_bt_intervals;
#else
    hibernate = this_thr->th.th_team_bt_intervals;
#endif /* KMP_ADJUST_BLOCKTIME */

    /* If the blocktime is nonzero, we want to make sure that we spin wait for
       the entirety of the specified #intervals, plus up to one interval more.
       This increment make certain that this thread doesn't go to sleep too
       soon.  */
    if (hibernate != 0)
      hibernate++;

    // Add in the current time value.
    hibernate += TCR_4(__kmp_global.g.g_time.dt.t_value);
    KF_TRACE(20, ("__kmp_wait_sleep: T#%d now=%d, hibernate=%d, intervals=%d\n",
                  th_gtid, __kmp_global.g.g_time.dt.t_value, hibernate,
                  hibernate - __kmp_global.g.g_time.dt.t_value));
#else
    if (__kmp_pause_status == kmp_soft_paused) {
      // Force immediate suspend
      hibernate_goal = KMP_NOW();
    } else
      hibernate_goal = KMP_NOW() + this_thr->th.th_team_bt_intervals;
    poll_count = 0;
    (void)poll_count;
#endif // KMP_USE_MONITOR
  }

  KMP_MB();

  // Main wait spin loop
  while (flag->notdone_check()) {
    kmp_task_team_t *task_team = NULL;
    if (__kmp_tasking_mode != tskm_immediate_exec) {
      task_team = this_thr->th.th_task_team;
      /* If the thread's task team pointer is NULL, it means one of 3 things:
         1) A newly-created thread is first being released by
         __kmp_fork_barrier(), and its task team has not been set up yet.
         2) All tasks have been executed to completion.
         3) Tasking is off for this region.  This could be because we are in a
         serialized region (perhaps the outer one), or else tasking was manually
         disabled (KMP_TASKING=0).  */
      if (task_team != NULL) {
        if (TCR_SYNC_4(task_team->tt.tt_active)) {
          if (KMP_TASKING_ENABLED(task_team)) {
            flag->execute_tasks(
                this_thr, th_gtid, final_spin,
                &tasks_completed USE_ITT_BUILD_ARG(itt_sync_obj), 0);
          } else
            this_thr->th.th_reap_state = KMP_SAFE_TO_REAP;
        } else {
          KMP_DEBUG_ASSERT(!KMP_MASTER_TID(this_thr->th.th_info.ds.ds_tid));
#if OMPT_SUPPORT
          // task-team is done now, other cases should be catched above
          if (final_spin && ompt_enabled.enabled)
            __ompt_implicit_task_end(this_thr, ompt_entry_state, tId);
#endif
          this_thr->th.th_task_team = NULL;
          this_thr->th.th_reap_state = KMP_SAFE_TO_REAP;
        }
      } else {
        this_thr->th.th_reap_state = KMP_SAFE_TO_REAP;
      } // if
    } // if

    KMP_FSYNC_SPIN_PREPARE(CCAST(void *, spin));
    if (TCR_4(__kmp_global.g.g_done)) {
      if (__kmp_global.g.g_abort)
        __kmp_abort_thread();
      break;
    }

    // If we are oversubscribed, or have waited a bit (and
    // KMP_LIBRARY=throughput), then yield
    KMP_YIELD_OVERSUB_ELSE_SPIN(spins, time);

#if KMP_STATS_ENABLED
    // Check if thread has been signalled to idle state
    // This indicates that the logical "join-barrier" has finished
    if (this_thr->th.th_stats->isIdle() &&
        KMP_GET_THREAD_STATE() == FORK_JOIN_BARRIER) {
      KMP_SET_THREAD_STATE(IDLE);
      KMP_PUSH_PARTITIONED_TIMER(OMP_idle);
    }
#endif
    // Check if the barrier surrounding this wait loop has been cancelled
    if (Cancellable) {
      kmp_team_t *team = this_thr->th.th_team;
      if (team && team->t.t_cancel_request == cancel_parallel)
        break;
    }

    // For hidden helper thread, if task_team is nullptr, it means the main
    // thread has not released the barrier. We cannot wait here because once the
    // main thread releases all children barriers, all hidden helper threads are
    // still sleeping. This leads to a problem that following configuration,
    // such as task team sync, will not be performed such that this thread does
    // not have task team. Usually it is not bad. However, a corner case is,
    // when the first task encountered is an untied task, the check in
    // __kmp_task_alloc will crash because it uses the task team pointer without
    // checking whether it is nullptr. It is probably under some kind of
    // assumption.
    if (task_team && KMP_HIDDEN_HELPER_WORKER_THREAD(th_gtid) &&
        !TCR_4(__kmp_hidden_helper_team_done)) {
      // If there is still hidden helper tasks to be executed, the hidden helper
      // thread will not enter a waiting status.
      if (KMP_ATOMIC_LD_ACQ(&__kmp_unexecuted_hidden_helper_tasks) == 0) {
        __kmp_hidden_helper_worker_thread_wait();
      }
      continue;
    }

    // Don't suspend if KMP_BLOCKTIME is set to "infinite"
    if (__kmp_dflt_blocktime == KMP_MAX_BLOCKTIME &&
        __kmp_pause_status != kmp_soft_paused)
      continue;

    // Don't suspend if there is a likelihood of new tasks being spawned.
    if ((task_team != NULL) && TCR_4(task_team->tt.tt_found_tasks))
      continue;

#if KMP_USE_MONITOR
    // If we have waited a bit more, fall asleep
    if (TCR_4(__kmp_global.g.g_time.dt.t_value) < hibernate)
      continue;
#else
    if (KMP_BLOCKING(hibernate_goal, poll_count++))
      continue;
#endif
    // Don't suspend if wait loop designated non-sleepable
    // in template parameters
    if (!Sleepable)
      continue;

    if (__kmp_dflt_blocktime == KMP_MAX_BLOCKTIME &&
        __kmp_pause_status != kmp_soft_paused)
      continue;

#if KMP_HAVE_MWAIT || KMP_HAVE_UMWAIT
    if (__kmp_mwait_enabled || __kmp_umwait_enabled) {
      KF_TRACE(50, ("__kmp_wait_sleep: T#%d using monitor/mwait\n", th_gtid));
      flag->mwait(th_gtid);
    } else {
#endif
      KF_TRACE(50, ("__kmp_wait_sleep: T#%d suspend time reached\n", th_gtid));
#if KMP_OS_UNIX
      if (final_spin)
        KMP_ATOMIC_ST_REL(&this_thr->th.th_blocking, false);
#endif
      flag->suspend(th_gtid);
#if KMP_OS_UNIX
      if (final_spin)
        KMP_ATOMIC_ST_REL(&this_thr->th.th_blocking, true);
#endif
#if KMP_HAVE_MWAIT || KMP_HAVE_UMWAIT
    }
#endif

    if (TCR_4(__kmp_global.g.g_done)) {
      if (__kmp_global.g.g_abort)
        __kmp_abort_thread();
      break;
    } else if (__kmp_tasking_mode != tskm_immediate_exec &&
               this_thr->th.th_reap_state == KMP_SAFE_TO_REAP) {
      this_thr->th.th_reap_state = KMP_NOT_SAFE_TO_REAP;
    }
    // TODO: If thread is done with work and times out, disband/free
  }

#if OMPT_SUPPORT
  ompt_state_t ompt_exit_state = this_thr->th.ompt_thread_info.state;
  if (ompt_enabled.enabled && ompt_exit_state != ompt_state_undefined) {
#if OMPT_OPTIONAL
    if (final_spin) {
      __ompt_implicit_task_end(this_thr, ompt_exit_state, tId);
      ompt_exit_state = this_thr->th.ompt_thread_info.state;
    }
#endif
    if (ompt_exit_state == ompt_state_idle) {
      this_thr->th.ompt_thread_info.state = ompt_state_overhead;
    }
  }
#endif
#if KMP_STATS_ENABLED
  // If we were put into idle state, pop that off the state stack
  if (KMP_GET_THREAD_STATE() == IDLE) {
    KMP_POP_PARTITIONED_TIMER();
    KMP_SET_THREAD_STATE(thread_state);
    this_thr->th.th_stats->resetIdleFlag();
  }
#endif

#if KMP_OS_UNIX
  if (final_spin)
    KMP_ATOMIC_ST_REL(&this_thr->th.th_blocking, false);
#endif
  KMP_FSYNC_SPIN_ACQUIRED(CCAST(void *, spin));
  if (Cancellable) {
    kmp_team_t *team = this_thr->th.th_team;
    if (team && team->t.t_cancel_request == cancel_parallel) {
      if (tasks_completed) {
        // undo the previous decrement of unfinished_threads so that the
        // thread can decrement at the join barrier with no problem
        kmp_task_team_t *task_team = this_thr->th.th_task_team;
        std::atomic<kmp_int32> *unfinished_threads =
            &(task_team->tt.tt_unfinished_threads);
        KMP_ATOMIC_INC(unfinished_threads);
      }
      return true;
    }
  }
  return false;
}

#if KMP_HAVE_MWAIT || KMP_HAVE_UMWAIT
// Set up a monitor on the flag variable causing the calling thread to wait in
// a less active state until the flag variable is modified.
template <class C>
static inline void __kmp_mwait_template(int th_gtid, C *flag) {
  KMP_TIME_DEVELOPER_PARTITIONED_BLOCK(USER_mwait);
  kmp_info_t *th = __kmp_threads[th_gtid];

  KF_TRACE(30, ("__kmp_mwait_template: T#%d enter for flag = %p\n", th_gtid,
                flag->get()));

  // User-level mwait is available
  KMP_DEBUG_ASSERT(__kmp_mwait_enabled || __kmp_umwait_enabled);

  __kmp_suspend_initialize_thread(th);
  __kmp_lock_suspend_mx(th);

  volatile void *spin = flag->get();
  void *cacheline = (void *)(kmp_uintptr_t(spin) & ~(CACHE_LINE - 1));

  if (!flag->done_check()) {
    // Mark thread as no longer active
    th->th.th_active = FALSE;
    if (th->th.th_active_in_pool) {
      th->th.th_active_in_pool = FALSE;
      KMP_ATOMIC_DEC(&__kmp_thread_pool_active_nth);
      KMP_DEBUG_ASSERT(TCR_4(__kmp_thread_pool_active_nth) >= 0);
    }
    flag->set_sleeping();
    KF_TRACE(50, ("__kmp_mwait_template: T#%d calling monitor\n", th_gtid));
#if KMP_HAVE_UMWAIT
    if (__kmp_umwait_enabled) {
      __kmp_umonitor(cacheline);
    }
#elif KMP_HAVE_MWAIT
    if (__kmp_mwait_enabled) {
      __kmp_mm_monitor(cacheline, 0, 0);
    }
#endif
    // To avoid a race, check flag between 'monitor' and 'mwait'. A write to
    // the address could happen after the last time we checked and before
    // monitoring started, in which case monitor can't detect the change.
    if (flag->done_check())
      flag->unset_sleeping();
    else {
      // if flag changes here, wake-up happens immediately
      TCW_PTR(th->th.th_sleep_loc, (void *)flag);
      th->th.th_sleep_loc_type = flag->get_type();
      __kmp_unlock_suspend_mx(th);
      KF_TRACE(50, ("__kmp_mwait_template: T#%d calling mwait\n", th_gtid));
#if KMP_HAVE_UMWAIT
      if (__kmp_umwait_enabled) {
        __kmp_umwait(1, 100); // to do: enable ctrl via hints, backoff counter
      }
#elif KMP_HAVE_MWAIT
      if (__kmp_mwait_enabled) {
        __kmp_mm_mwait(0, __kmp_mwait_hints);
      }
#endif
      KF_TRACE(50, ("__kmp_mwait_template: T#%d mwait done\n", th_gtid));
      __kmp_lock_suspend_mx(th);
      // Clean up sleep info; doesn't matter how/why this thread stopped waiting
      if (flag->is_sleeping())
        flag->unset_sleeping();
      TCW_PTR(th->th.th_sleep_loc, NULL);
      th->th.th_sleep_loc_type = flag_unset;
    }
    // Mark thread as active again
    th->th.th_active = TRUE;
    if (TCR_4(th->th.th_in_pool)) {
      KMP_ATOMIC_INC(&__kmp_thread_pool_active_nth);
      th->th.th_active_in_pool = TRUE;
    }
  } // Drop out to main wait loop to check flag, handle tasks, etc.
  __kmp_unlock_suspend_mx(th);
  KF_TRACE(30, ("__kmp_mwait_template: T#%d exit\n", th_gtid));
}
#endif // KMP_HAVE_MWAIT || KMP_HAVE_UMWAIT

/* Release any threads specified as waiting on the flag by releasing the flag
   and resume the waiting thread if indicated by the sleep bit(s). A thread that
   calls __kmp_wait_template must call this function to wake up the potentially
   sleeping thread and prevent deadlocks!  */
template <class C> static inline void __kmp_release_template(C *flag) {
#ifdef KMP_DEBUG
  int gtid = TCR_4(__kmp_init_gtid) ? __kmp_get_gtid() : -1;
#endif
  KF_TRACE(20, ("__kmp_release: T#%d releasing flag(%x)\n", gtid, flag->get()));
  KMP_DEBUG_ASSERT(flag->get());
  KMP_FSYNC_RELEASING(flag->get_void_p());

  flag->internal_release();

  KF_TRACE(100, ("__kmp_release: T#%d set new spin=%d\n", gtid, flag->get(),
                 flag->load()));

  if (__kmp_dflt_blocktime != KMP_MAX_BLOCKTIME) {
    // Only need to check sleep stuff if infinite block time not set.
    // Are *any* threads waiting on flag sleeping?
    if (flag->is_any_sleeping()) {
      for (unsigned int i = 0; i < flag->get_num_waiters(); ++i) {
        // if sleeping waiter exists at i, sets current_waiter to i inside flag
        kmp_info_t *waiter = flag->get_waiter(i);
        if (waiter) {
          int wait_gtid = waiter->th.th_info.ds.ds_gtid;
          // Wake up thread if needed
          KF_TRACE(50, ("__kmp_release: T#%d waking up thread T#%d since sleep "
                        "flag(%p) set\n",
                        gtid, wait_gtid, flag->get()));
          flag->resume(wait_gtid); // unsets flag's current_waiter when done
        }
      }
    }
  }
}

template <bool Cancellable, bool Sleepable>
class kmp_flag_32 : public kmp_flag_atomic<kmp_uint32, flag32, Sleepable> {
public:
  kmp_flag_32(std::atomic<kmp_uint32> *p)
      : kmp_flag_atomic<kmp_uint32, flag32, Sleepable>(p) {}
  kmp_flag_32(std::atomic<kmp_uint32> *p, kmp_info_t *thr)
      : kmp_flag_atomic<kmp_uint32, flag32, Sleepable>(p, thr) {}
  kmp_flag_32(std::atomic<kmp_uint32> *p, kmp_uint32 c)
      : kmp_flag_atomic<kmp_uint32, flag32, Sleepable>(p, c) {}
  void suspend(int th_gtid) { __kmp_suspend_32(th_gtid, this); }
#if KMP_HAVE_MWAIT || KMP_HAVE_UMWAIT
  void mwait(int th_gtid) { __kmp_mwait_32(th_gtid, this); }
#endif
  void resume(int th_gtid) { __kmp_resume_32(th_gtid, this); }
  int execute_tasks(kmp_info_t *this_thr, kmp_int32 gtid, int final_spin,
                    int *thread_finished USE_ITT_BUILD_ARG(void *itt_sync_obj),
                    kmp_int32 is_constrained) {
    return __kmp_execute_tasks_32(
        this_thr, gtid, this, final_spin,
        thread_finished USE_ITT_BUILD_ARG(itt_sync_obj), is_constrained);
  }
  bool wait(kmp_info_t *this_thr,
            int final_spin USE_ITT_BUILD_ARG(void *itt_sync_obj)) {
    if (final_spin)
      return __kmp_wait_template<kmp_flag_32, TRUE, Cancellable, Sleepable>(
          this_thr, this USE_ITT_BUILD_ARG(itt_sync_obj));
    else
      return __kmp_wait_template<kmp_flag_32, FALSE, Cancellable, Sleepable>(
          this_thr, this USE_ITT_BUILD_ARG(itt_sync_obj));
  }
  void release() { __kmp_release_template(this); }
  flag_type get_ptr_type() { return flag32; }
};

template <bool Cancellable, bool Sleepable>
class kmp_flag_64 : public kmp_flag_native<kmp_uint64, flag64, Sleepable> {
public:
  kmp_flag_64(volatile kmp_uint64 *p)
      : kmp_flag_native<kmp_uint64, flag64, Sleepable>(p) {}
  kmp_flag_64(volatile kmp_uint64 *p, kmp_info_t *thr)
      : kmp_flag_native<kmp_uint64, flag64, Sleepable>(p, thr) {}
  kmp_flag_64(volatile kmp_uint64 *p, kmp_uint64 c)
      : kmp_flag_native<kmp_uint64, flag64, Sleepable>(p, c) {}
  kmp_flag_64(volatile kmp_uint64 *p, kmp_uint64 c, std::atomic<bool> *loc)
      : kmp_flag_native<kmp_uint64, flag64, Sleepable>(p, c, loc) {}
  void suspend(int th_gtid) { __kmp_suspend_64(th_gtid, this); }
#if KMP_HAVE_MWAIT || KMP_HAVE_UMWAIT
  void mwait(int th_gtid) { __kmp_mwait_64(th_gtid, this); }
#endif
  void resume(int th_gtid) { __kmp_resume_64(th_gtid, this); }
  int execute_tasks(kmp_info_t *this_thr, kmp_int32 gtid, int final_spin,
                    int *thread_finished USE_ITT_BUILD_ARG(void *itt_sync_obj),
                    kmp_int32 is_constrained) {
    return __kmp_execute_tasks_64(
        this_thr, gtid, this, final_spin,
        thread_finished USE_ITT_BUILD_ARG(itt_sync_obj), is_constrained);
  }
  bool wait(kmp_info_t *this_thr,
            int final_spin USE_ITT_BUILD_ARG(void *itt_sync_obj)) {
    if (final_spin)
      return __kmp_wait_template<kmp_flag_64, TRUE, Cancellable, Sleepable>(
          this_thr, this USE_ITT_BUILD_ARG(itt_sync_obj));
    else
      return __kmp_wait_template<kmp_flag_64, FALSE, Cancellable, Sleepable>(
          this_thr, this USE_ITT_BUILD_ARG(itt_sync_obj));
  }
  void release() { __kmp_release_template(this); }
  flag_type get_ptr_type() { return flag64; }
};

template <bool Cancellable, bool Sleepable>
class kmp_atomic_flag_64
    : public kmp_flag_atomic<kmp_uint64, atomic_flag64, Sleepable> {
public:
  kmp_atomic_flag_64(std::atomic<kmp_uint64> *p)
      : kmp_flag_atomic<kmp_uint64, atomic_flag64, Sleepable>(p) {}
  kmp_atomic_flag_64(std::atomic<kmp_uint64> *p, kmp_info_t *thr)
      : kmp_flag_atomic<kmp_uint64, atomic_flag64, Sleepable>(p, thr) {}
  kmp_atomic_flag_64(std::atomic<kmp_uint64> *p, kmp_uint64 c)
      : kmp_flag_atomic<kmp_uint64, atomic_flag64, Sleepable>(p, c) {}
  kmp_atomic_flag_64(std::atomic<kmp_uint64> *p, kmp_uint64 c,
                     std::atomic<bool> *loc)
      : kmp_flag_atomic<kmp_uint64, atomic_flag64, Sleepable>(p, c, loc) {}
  void suspend(int th_gtid) { __kmp_atomic_suspend_64(th_gtid, this); }
  void mwait(int th_gtid) { __kmp_atomic_mwait_64(th_gtid, this); }
  void resume(int th_gtid) { __kmp_atomic_resume_64(th_gtid, this); }
  int execute_tasks(kmp_info_t *this_thr, kmp_int32 gtid, int final_spin,
                    int *thread_finished USE_ITT_BUILD_ARG(void *itt_sync_obj),
                    kmp_int32 is_constrained) {
    return __kmp_atomic_execute_tasks_64(
        this_thr, gtid, this, final_spin,
        thread_finished USE_ITT_BUILD_ARG(itt_sync_obj), is_constrained);
  }
  bool wait(kmp_info_t *this_thr,
            int final_spin USE_ITT_BUILD_ARG(void *itt_sync_obj)) {
    if (final_spin)
      return __kmp_wait_template<kmp_atomic_flag_64, TRUE, Cancellable,
                                 Sleepable>(
          this_thr, this USE_ITT_BUILD_ARG(itt_sync_obj));
    else
      return __kmp_wait_template<kmp_atomic_flag_64, FALSE, Cancellable,
                                 Sleepable>(
          this_thr, this USE_ITT_BUILD_ARG(itt_sync_obj));
  }
  void release() { __kmp_release_template(this); }
  flag_type get_ptr_type() { return atomic_flag64; }
};

// Hierarchical 64-bit on-core barrier instantiation
class kmp_flag_oncore : public kmp_flag_native<kmp_uint64, flag_oncore, false> {
  kmp_uint32 offset; /**< Portion of flag of interest for an operation. */
  bool flag_switch; /**< Indicates a switch in flag location. */
  enum barrier_type bt; /**< Barrier type. */
  kmp_info_t *this_thr; /**< Thread to redirect to different flag location. */
#if USE_ITT_BUILD
  void *itt_sync_obj; /**< ITT object to pass to new flag location. */
#endif
  unsigned char &byteref(volatile kmp_uint64 *loc, size_t offset) {
    return (RCAST(unsigned char *, CCAST(kmp_uint64 *, loc)))[offset];
  }

public:
  kmp_flag_oncore(volatile kmp_uint64 *p)
      : kmp_flag_native<kmp_uint64, flag_oncore, false>(p), flag_switch(false) {
  }
  kmp_flag_oncore(volatile kmp_uint64 *p, kmp_uint32 idx)
      : kmp_flag_native<kmp_uint64, flag_oncore, false>(p), offset(idx),
        flag_switch(false),
        bt(bs_last_barrier) USE_ITT_BUILD_ARG(itt_sync_obj(nullptr)) {}
  kmp_flag_oncore(volatile kmp_uint64 *p, kmp_uint64 c, kmp_uint32 idx,
                  enum barrier_type bar_t,
                  kmp_info_t *thr USE_ITT_BUILD_ARG(void *itt))
      : kmp_flag_native<kmp_uint64, flag_oncore, false>(p, c), offset(idx),
        flag_switch(false), bt(bar_t),
        this_thr(thr) USE_ITT_BUILD_ARG(itt_sync_obj(itt)) {}
  virtual ~kmp_flag_oncore() override {}
  void *operator new(size_t size) { return __kmp_allocate(size); }
  void operator delete(void *p) { __kmp_free(p); }
  bool done_check_val(kmp_uint64 old_loc) override {
    return byteref(&old_loc, offset) == checker;
  }
  bool done_check() override { return done_check_val(*get()); }
  bool notdone_check() override {
    // Calculate flag_switch
    if (this_thr->th.th_bar[bt].bb.wait_flag == KMP_BARRIER_SWITCH_TO_OWN_FLAG)
      flag_switch = true;
    if (byteref(get(), offset) != 1 && !flag_switch)
      return true;
    else if (flag_switch) {
      this_thr->th.th_bar[bt].bb.wait_flag = KMP_BARRIER_SWITCHING;
      kmp_flag_64<> flag(&this_thr->th.th_bar[bt].bb.b_go,
                         (kmp_uint64)KMP_BARRIER_STATE_BUMP);
      __kmp_wait_64(this_thr, &flag, TRUE USE_ITT_BUILD_ARG(itt_sync_obj));
    }
    return false;
  }
  void internal_release() {
    // Other threads can write their own bytes simultaneously.
    if (__kmp_dflt_blocktime == KMP_MAX_BLOCKTIME) {
      byteref(get(), offset) = 1;
    } else {
      kmp_uint64 mask = 0;
      byteref(&mask, offset) = 1;
      KMP_TEST_THEN_OR64(get(), mask);
    }
  }
  void wait(kmp_info_t *this_thr, int final_spin) {
    if (final_spin)
      __kmp_wait_template<kmp_flag_oncore, TRUE>(
          this_thr, this USE_ITT_BUILD_ARG(itt_sync_obj));
    else
      __kmp_wait_template<kmp_flag_oncore, FALSE>(
          this_thr, this USE_ITT_BUILD_ARG(itt_sync_obj));
  }
  void release() { __kmp_release_template(this); }
  void suspend(int th_gtid) { __kmp_suspend_oncore(th_gtid, this); }
#if KMP_HAVE_MWAIT || KMP_HAVE_UMWAIT
  void mwait(int th_gtid) { __kmp_mwait_oncore(th_gtid, this); }
#endif
  void resume(int th_gtid) { __kmp_resume_oncore(th_gtid, this); }
  int execute_tasks(kmp_info_t *this_thr, kmp_int32 gtid, int final_spin,
                    int *thread_finished USE_ITT_BUILD_ARG(void *itt_sync_obj),
                    kmp_int32 is_constrained) {
#if OMPD_SUPPORT
    int ret = __kmp_execute_tasks_oncore(
        this_thr, gtid, this, final_spin,
        thread_finished USE_ITT_BUILD_ARG(itt_sync_obj), is_constrained);
    if (ompd_state & OMPD_ENABLE_BP)
      ompd_bp_task_end();
    return ret;
#else
    return __kmp_execute_tasks_oncore(
        this_thr, gtid, this, final_spin,
        thread_finished USE_ITT_BUILD_ARG(itt_sync_obj), is_constrained);
#endif
  }
  enum barrier_type get_bt() { return bt; }
  flag_type get_ptr_type() { return flag_oncore; }
};

static inline void __kmp_null_resume_wrapper(kmp_info_t *thr) {
  int gtid = __kmp_gtid_from_thread(thr);
  void *flag = CCAST(void *, thr->th.th_sleep_loc);
  flag_type type = thr->th.th_sleep_loc_type;
  if (!flag)
    return;
  // Attempt to wake up a thread: examine its type and call appropriate template
  switch (type) {
  case flag32:
    __kmp_resume_32(gtid, RCAST(kmp_flag_32<> *, flag));
    break;
  case flag64:
    __kmp_resume_64(gtid, RCAST(kmp_flag_64<> *, flag));
    break;
  case atomic_flag64:
    __kmp_atomic_resume_64(gtid, RCAST(kmp_atomic_flag_64<> *, flag));
    break;
  case flag_oncore:
    __kmp_resume_oncore(gtid, RCAST(kmp_flag_oncore *, flag));
    break;
#ifdef KMP_DEBUG
  case flag_unset:
    KF_TRACE(100, ("__kmp_null_resume_wrapper: flag type %d is unset\n", type));
    break;
  default:
    KF_TRACE(100, ("__kmp_null_resume_wrapper: flag type %d does not match any "
                   "known flag type\n",
                   type));
#endif
  }
}

/*!
@}
*/

#endif // KMP_WAIT_RELEASE_H
