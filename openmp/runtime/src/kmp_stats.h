#ifndef KMP_STATS_H
#define KMP_STATS_H

/** @file kmp_stats.h
 * Functions for collecting statistics.
 */


//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//

#include "kmp_config.h"

#if KMP_STATS_ENABLED
/*
 * Statistics accumulator.
 * Accumulates number of samples and computes min, max, mean, standard deviation on the fly.
 *
 * Online variance calculation algorithm from http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#On-line_algorithm
 */

#include <limits>
#include <math.h>
#include <vector>
#include <string>
#include <stdint.h>
#include <new> // placement new
#include "kmp_stats_timing.h"

/*
 * Enable developer statistics here if you want them. They are more detailed than is useful for application characterisation and
 * are intended for the runtime library developer.
 */
// #define KMP_DEVELOPER_STATS 1

/*!
 * @ingroup STATS_GATHERING
 * \brief flags to describe the statistic (timer or counter)
 *
 */
enum stats_flags_e {
    noTotal      = 1<<0,     //!< do not show a TOTAL_aggregation for this statistic
    onlyInMaster = 1<<1,     //!< statistic is valid only for master
    noUnits      = 1<<2,     //!< statistic doesn't need units printed next to it in output
    notInMaster  = 1<<3,     //!< statistic is valid only for non-master threads
    logEvent     = 1<<4      //!< statistic can be logged on the event timeline when KMP_STATS_EVENTS is on (valid only for timers)
};

/*!
 * @ingroup STATS_GATHERING
 * \brief the states which a thread can be in
 *
 */
enum stats_state_e {
    IDLE,
    SERIAL_REGION,
    FORK_JOIN_BARRIER,
    PLAIN_BARRIER,
    TASKWAIT,
    TASKYIELD,
    TASKGROUP,
    IMPLICIT_TASK,
    EXPLICIT_TASK
};

/*!
 * \brief Add new counters under KMP_FOREACH_COUNTER() macro in kmp_stats.h
 *
 * @param macro a user defined macro that takes three arguments - macro(COUNTER_NAME, flags, arg)
 * @param arg a user defined argument to send to the user defined macro
 *
 * \details A counter counts the occurrence of some event.
 * Each thread accumulates its own count, at the end of execution the counts are aggregated treating each thread
 * as a separate measurement. (Unless onlyInMaster is set, in which case there's only a single measurement).
 * The min,mean,max are therefore the values for the threads.
 * Adding the counter here and then putting a KMP_BLOCK_COUNTER(name) at the point you want to count is all you need to do.
 * All of the tables and printing is generated from this macro.
 * Format is "macro(name, flags, arg)"
 *
 * @ingroup STATS_GATHERING
 */
#define KMP_FOREACH_COUNTER(macro, arg)                         \
    macro (OMP_PARALLEL, stats_flags_e::onlyInMaster | stats_flags_e::noTotal, arg) \
    macro (OMP_NESTED_PARALLEL, 0, arg)                         \
    macro (OMP_FOR_static, 0, arg)                              \
    macro (OMP_FOR_dynamic, 0, arg)                             \
    macro (OMP_DISTRIBUTE, 0, arg)                              \
    macro (OMP_BARRIER, 0, arg)                                 \
    macro (OMP_CRITICAL,0, arg)                                 \
    macro (OMP_SINGLE, 0, arg)                                  \
    macro (OMP_MASTER, 0, arg)                                  \
    macro (OMP_TEAMS, 0, arg)                                   \
    macro (OMP_set_lock, 0, arg)                                \
    macro (OMP_test_lock, 0, arg)                               \
    macro (REDUCE_wait, 0, arg)                                 \
    macro (REDUCE_nowait, 0, arg)                               \
    macro (OMP_TASKYIELD, 0, arg)                               \
    macro (OMP_TASKLOOP, 0, arg)                                \
    macro (TASK_executed, 0, arg)                               \
    macro (TASK_cancelled, 0, arg)                              \
    macro (TASK_stolen, 0, arg)                                 \
    macro (LAST,0,arg)

/*!
 * \brief Add new timers under KMP_FOREACH_TIMER() macro in kmp_stats.h
 *
 * @param macro a user defined macro that takes three arguments - macro(TIMER_NAME, flags, arg)
 * @param arg a user defined argument to send to the user defined macro
 *
 * \details A timer collects multiple samples of some count in each thread and then finally aggregates alll of the samples from all of the threads.
 * For most timers the printing code also provides an aggregation over the thread totals. These are printed as TOTAL_foo.
 * The count is normally a time (in ticks), hence the name "timer". (But can be any value, so we use this for "number of arguments passed to fork"
 * as well).
 * For timers the threads are not significant, it's the individual observations that count, so the statistics are at that level.
 * Format is "macro(name, flags, arg)"
 *
 * @ingroup STATS_GATHERING2
 */
#define KMP_FOREACH_TIMER(macro, arg)                              \
    macro (OMP_worker_thread_life, 0, arg)                         \
    macro (FOR_static_scheduling, 0, arg)                          \
    macro (FOR_dynamic_scheduling, 0, arg)                         \
    macro (OMP_critical,  0, arg)                                  \
    macro (OMP_critical_wait,  0, arg)                             \
    macro (OMP_single,    0, arg)                                  \
    macro (OMP_master,    0, arg)                                  \
    macro (OMP_idle, 0, arg)                                       \
    macro (OMP_plain_barrier, 0, arg)                              \
    macro (OMP_fork_join_barrier, 0, arg)                          \
    macro (OMP_parallel, 0, arg)                                   \
    macro (OMP_task_immediate, 0, arg)                             \
    macro (OMP_task_taskwait, 0, arg)                              \
    macro (OMP_task_taskyield, 0, arg)                             \
    macro (OMP_task_taskgroup, 0, arg)                             \
    macro (OMP_task_join_bar, 0, arg)                              \
    macro (OMP_task_plain_bar, 0, arg)                             \
    macro (OMP_serial, 0, arg)                                     \
    macro (OMP_taskloop_scheduling, 0, arg)                        \
    macro (OMP_set_numthreads,    stats_flags_e::noUnits | stats_flags_e::noTotal, arg) \
    macro (OMP_PARALLEL_args,     stats_flags_e::noUnits | stats_flags_e::noTotal, arg) \
    macro (FOR_static_iterations, stats_flags_e::noUnits | stats_flags_e::noTotal, arg) \
    macro (FOR_dynamic_iterations,stats_flags_e::noUnits | stats_flags_e::noTotal, arg) \
    KMP_FOREACH_DEVELOPER_TIMER(macro, arg)                             \
    macro (LAST,0, arg)


// OMP_start_end          -- Time from when OpenMP is initialized until the stats are printed at exit
// OMP_serial             -- Thread zero time executing serial code
// OMP_work               -- Elapsed time in code dispatched by a fork (measured in the thread)
// OMP_barrier            -- Time at "real" barriers (includes task time)
// FOR_static_scheduling  -- Time spent doing scheduling for a static "for"
// FOR_dynamic_scheduling -- Time spent doing scheduling for a dynamic "for"
// OMP_idle               -- Worker threads time spent waiting for inclusion in a parallel region
// OMP_plain_barrier      -- Time spent in a barrier construct
// OMP_fork_join_barrier  -- Time spent in a the fork-join barrier surrounding a parallel region
// OMP_parallel           -- Time spent inside a parallel construct
// OMP_task_immediate     -- Time spent executing non-deferred tasks
// OMP_task_taskwait      -- Time spent executing tasks inside a taskwait construct
// OMP_task_taskyield     -- Time spent executing tasks inside a taskyield construct
// OMP_task_taskgroup     -- Time spent executing tasks inside a taskygroup construct
// OMP_task_join_bar      -- Time spent executing tasks inside a join barrier
// OMP_task_plain_bar     -- Time spent executing tasks inside a barrier construct
// OMP_single             -- Time spent executing a "single" region
// OMP_master             -- Time spent executing a "master" region
// OMP_set_numthreads     -- Values passed to omp_set_num_threads
// OMP_PARALLEL_args      -- Number of arguments passed to a parallel region
// FOR_static_iterations  -- Number of available parallel chunks of work in a static for
// FOR_dynamic_iterations -- Number of available parallel chunks of work in a dynamic for
//                           Both adjust for any chunking, so if there were an iteration count of 20 but a chunk size of 10, we'd record 2.

#if (KMP_DEVELOPER_STATS)
// Timers which are of interest to runtime library developers, not end users.
// These have to be explicitly enabled in addition to the other stats.

// KMP_fork_barrier       -- time in __kmp_fork_barrier
// KMP_join_barrier       -- time in __kmp_join_barrier
// KMP_barrier            -- time in __kmp_barrier
// KMP_end_split_barrier  -- time in __kmp_end_split_barrier
// KMP_setup_icv_copy     -- time in __kmp_setup_icv_copy
// KMP_icv_copy           -- start/stop timer for any ICV copying
// KMP_linear_gather      -- time in __kmp_linear_barrier_gather
// KMP_linear_release     -- time in __kmp_linear_barrier_release
// KMP_tree_gather        -- time in __kmp_tree_barrier_gather
// KMP_tree_release       -- time in __kmp_tree_barrier_release
// KMP_hyper_gather       -- time in __kmp_hyper_barrier_gather
// KMP_hyper_release      -- time in __kmp_hyper_barrier_release
# define KMP_FOREACH_DEVELOPER_TIMER(macro, arg)                        \
    macro (KMP_fork_call, 0, arg)                                       \
    macro (KMP_join_call, 0, arg)                                       \
    macro (KMP_fork_barrier, stats_flags_e::logEvent, arg)              \
    macro (KMP_join_barrier, stats_flags_e::logEvent, arg)              \
    macro (KMP_barrier, 0, arg)                                         \
    macro (KMP_end_split_barrier, 0, arg)                               \
    macro (KMP_hier_gather, 0, arg)                                     \
    macro (KMP_hier_release, 0, arg)                                    \
    macro (KMP_hyper_gather,  stats_flags_e::logEvent, arg)             \
    macro (KMP_hyper_release,  stats_flags_e::logEvent, arg)            \
    macro (KMP_linear_gather, 0, arg)                                   \
    macro (KMP_linear_release, 0, arg)                                  \
    macro (KMP_tree_gather, 0, arg)                                     \
    macro (KMP_tree_release, 0, arg)                                    \
    macro (USER_master_invoke, stats_flags_e::logEvent, arg)            \
    macro (USER_worker_invoke, stats_flags_e::logEvent, arg)            \
    macro (USER_resume, stats_flags_e::logEvent, arg)                   \
    macro (USER_suspend, stats_flags_e::logEvent, arg)                  \
    macro (USER_launch_thread_loop, stats_flags_e::logEvent, arg)       \
    macro (KMP_allocate_team, 0, arg)                                   \
    macro (KMP_setup_icv_copy, 0, arg)                                  \
    macro (USER_icv_copy, 0, arg)
#else
# define KMP_FOREACH_DEVELOPER_TIMER(macro, arg)
#endif

/*!
 * \brief Add new explicit timers under KMP_FOREACH_EXPLICIT_TIMER() macro.
 *
 * @param macro a user defined macro that takes three arguments - macro(TIMER_NAME, flags, arg)
 * @param arg a user defined argument to send to the user defined macro
 *
 * \warning YOU MUST HAVE THE SAME NAMED TIMER UNDER KMP_FOREACH_TIMER() OR ELSE BAD THINGS WILL HAPPEN!
 *
 * \details Explicit timers are ones where we need to allocate a timer itself (as well as the accumulated timing statistics).
 * We allocate these on a per-thread basis, and explicitly start and stop them.
 * Block timers just allocate the timer itself on the stack, and use the destructor to notice block exit; they don't
 * need to be defined here.
 * The name here should be the same as that of a timer above.
 *
 * @ingroup STATS_GATHERING
*/
#define KMP_FOREACH_EXPLICIT_TIMER(macro, arg)     \
    macro(OMP_worker_thread_life, 0, arg)          \
    macro(FOR_static_scheduling, 0, arg)           \
    macro(FOR_dynamic_scheduling, 0, arg)          \
    macro(OMP_critical, 0, arg)                    \
    macro(OMP_critical_wait, 0, arg)               \
    macro(OMP_single, 0, arg)                      \
    macro(OMP_master, 0, arg)                      \
    macro(OMP_idle, 0, arg)                        \
    macro(OMP_plain_barrier, 0, arg)               \
    macro(OMP_fork_join_barrier, 0, arg)           \
    macro(OMP_parallel, 0, arg)                    \
    macro(OMP_task_immediate, 0, arg)              \
    macro(OMP_task_taskwait, 0, arg)               \
    macro(OMP_task_taskyield, 0, arg)              \
    macro(OMP_task_taskgroup, 0, arg)              \
    macro(OMP_task_join_bar, 0, arg)               \
    macro(OMP_task_plain_bar, 0, arg)              \
    macro(OMP_serial, 0, arg)                      \
    macro(OMP_taskloop_scheduling, 0, arg)         \
    KMP_FOREACH_EXPLICIT_DEVELOPER_TIMER(macro,arg)     \
    macro(LAST, 0, arg)

#if (KMP_DEVELOPER_STATS)
# define KMP_FOREACH_EXPLICIT_DEVELOPER_TIMER(macro, arg)               \
    macro(USER_launch_thread_loop, stats_flags_e::logEvent, arg)
#else
# define KMP_FOREACH_EXPLICIT_DEVELOPER_TIMER(macro, arg)
#endif

#define ENUMERATE(name,ignore,prefix) prefix##name,
enum timer_e {
    KMP_FOREACH_TIMER(ENUMERATE, TIMER_)
};

enum explicit_timer_e {
    KMP_FOREACH_EXPLICIT_TIMER(ENUMERATE, EXPLICIT_TIMER_)
};

enum counter_e {
    KMP_FOREACH_COUNTER(ENUMERATE, COUNTER_)
};
#undef ENUMERATE

class timerPair {
    explicit_timer_e timer_index;
    timer_e timer;
 public:
    timerPair(explicit_timer_e ti, timer_e t) : timer_index(ti), timer(t) {}
    inline explicit_timer_e get_index() const { return timer_index; }
    inline timer_e get_timer() const { return timer; }
    bool operator==(const timerPair & rhs) {
        return this->get_index() == rhs.get_index();
    }
    bool operator!=(const timerPair & rhs) {
        return !(*this == rhs);
    }
};

class statistic
{
    double   minVal;
    double   maxVal;
    double   meanVal;
    double   m2;
    uint64_t sampleCount;

 public:
    statistic() { reset(); }
    statistic (statistic const &o): minVal(o.minVal), maxVal(o.maxVal), meanVal(o.meanVal), m2(o.m2), sampleCount(o.sampleCount) {}

    double   getMin()   const { return minVal; }
    double   getMean()  const { return meanVal; }
    double   getMax()   const { return maxVal; }
    uint64_t getCount() const { return sampleCount; }
    double   getSD()    const { return sqrt(m2/sampleCount); }
    double   getTotal() const { return sampleCount*meanVal; }

    void reset()
    {
        minVal =  std::numeric_limits<double>::max();
        maxVal = -std::numeric_limits<double>::max();
        meanVal= 0.0;
        m2     = 0.0;
        sampleCount = 0;
    }
    void addSample(double sample);
    void scale    (double factor);
    void scaleDown(double f)  { scale (1./f); }
    statistic & operator+= (statistic const & other);

    std::string format(char unit, bool total=false) const;
};

struct statInfo
{
    const char * name;
    uint32_t     flags;
};

class timeStat : public statistic
{
    static statInfo timerInfo[];

 public:
    timeStat() : statistic() {}
    static const char * name(timer_e e) { return timerInfo[e].name; }
    static bool  noTotal    (timer_e e) { return timerInfo[e].flags & stats_flags_e::noTotal;      }
    static bool  masterOnly (timer_e e) { return timerInfo[e].flags & stats_flags_e::onlyInMaster; }
    static bool  workerOnly (timer_e e) { return timerInfo[e].flags & stats_flags_e::notInMaster;  }
    static bool  noUnits    (timer_e e) { return timerInfo[e].flags & stats_flags_e::noUnits;      }
    static bool  logEvent   (timer_e e) { return timerInfo[e].flags & stats_flags_e::logEvent;     }
    static void  clearEventFlags()      {
        for(int i=0;i<TIMER_LAST;i++) {
            timerInfo[i].flags &= (~(stats_flags_e::logEvent));
        }
    }
};

// Where we need explicitly to start and end the timer, this version can be used
// Since these timers normally aren't nicely scoped, so don't have a good place to live
// on the stack of the thread, they're more work to use.
class explicitTimer
{
    timeStat * stat;
    tsc_tick_count startTime;
    tsc_tick_count pauseStartTime;
    tsc_tick_count::tsc_interval_t totalPauseTime;

 public:
    explicitTimer () : stat(0), startTime(0), pauseStartTime(0), totalPauseTime() { }
    explicitTimer (timeStat * s) : stat(s), startTime(), pauseStartTime(0), totalPauseTime() { }

    void setStat (timeStat *s) { stat = s; }
    void start(timer_e timerEnumValue);
    void pause() { pauseStartTime = tsc_tick_count::now(); }
    void resume() { totalPauseTime += (tsc_tick_count::now() - pauseStartTime); }
    void stop(timer_e timerEnumValue);
    void reset() { startTime = 0; pauseStartTime = 0; totalPauseTime = 0; }
};

// Where all you need is to time a block, this is enough.
// (It avoids the need to have an explicit end, leaving the scope suffices.)
class blockTimer : public explicitTimer
{
    timer_e timerEnumValue;
 public:
    blockTimer (timeStat * s, timer_e newTimerEnumValue) : timerEnumValue(newTimerEnumValue), explicitTimer(s) { start(timerEnumValue); }
    ~blockTimer() { stop(timerEnumValue); }
};

// Where you need to partition a threads clock ticks into separate states
// e.g., a partitionedTimers class with two timers of EXECUTING_TASK, and
//   DOING_NOTHING would render these conditions:
//   time(EXECUTING_TASK) + time(DOING_NOTHING) = total time thread is alive
//   No clock tick in the EXECUTING_TASK is a member of DOING_NOTHING and vice versa
class partitionedTimers
{
 private:
    explicitTimer* timers[EXPLICIT_TIMER_LAST+1];
    std::vector<timerPair> timer_stack;
 public:
    partitionedTimers();
    void add_timer(explicit_timer_e timer_index, explicitTimer* timer_pointer);
    void init(timerPair timer_index);
    void push(timerPair timer_index);
    void pop();
    void windup();
};

// Special wrapper around the partioned timers to aid timing code blocks
// It avoids the need to have an explicit end, leaving the scope suffices.
class blockPartitionedTimer
{
    partitionedTimers* part_timers;
    timerPair timer_pair;
 public:
    blockPartitionedTimer(partitionedTimers* pt, timerPair tp) : part_timers(pt), timer_pair(tp) { part_timers->push(timer_pair); }
   ~blockPartitionedTimer() { part_timers->pop(); }
};

// Special wrapper around the thread state to aid in keeping state in code blocks
// It avoids the need to have an explicit end, leaving the scope suffices.
class blockThreadState
{
    stats_state_e* state_pointer;
    stats_state_e  old_state;
 public:
    blockThreadState(stats_state_e* thread_state_pointer, stats_state_e new_state) : state_pointer(thread_state_pointer), old_state(*thread_state_pointer) {
        *state_pointer = new_state;
    }
   ~blockThreadState() { *state_pointer = old_state;  }
};

// If all you want is a count, then you can use this...
// The individual per-thread counts will be aggregated into a statistic at program exit.
class counter
{
    uint64_t value;
    static const statInfo counterInfo[];

 public:
    counter() : value(0) {}
    void increment() { value++; }
    uint64_t getValue() const { return value; }
    void reset() { value = 0; }
    static const char * name(counter_e e) { return counterInfo[e].name; }
    static bool  masterOnly (counter_e e) { return counterInfo[e].flags & stats_flags_e::onlyInMaster; }
};

/* ****************************************************************
    Class to implement an event

    There are four components to an event: start time, stop time
    nest_level, and timer_name.
    The start and stop time should be obvious (recorded in clock ticks).
    The nest_level relates to the bar width in the timeline graph.
    The timer_name is used to determine which timer event triggered this event.

    the interface to this class is through four read-only operations:
    1) getStart()     -- returns the start time as 64 bit integer
    2) getStop()      -- returns the stop time as 64 bit integer
    3) getNestLevel() -- returns the nest level of the event
    4) getTimerName() -- returns the timer name that triggered event

    *MORE ON NEST_LEVEL*
    The nest level is used in the bar graph that represents the timeline.
    Its main purpose is for showing how events are nested inside eachother.
    For example, say events, A, B, and C are recorded.  If the timeline
    looks like this:

Begin -------------------------------------------------------------> Time
         |    |          |        |          |              |
         A    B          C        C          B              A
       start start     start     end        end            end

       Then A, B, C will have a nest level of 1, 2, 3 respectively.
       These values are then used to calculate the barwidth so you can
       see that inside A, B has occurred, and inside B, C has occurred.
       Currently, this is shown with A's bar width being larger than B's
       bar width, and B's bar width being larger than C's bar width.

**************************************************************** */
class kmp_stats_event {
    uint64_t start;
    uint64_t stop;
    int nest_level;
    timer_e timer_name;
 public:
    kmp_stats_event() : start(0), stop(0), nest_level(0), timer_name(TIMER_LAST) {}
    kmp_stats_event(uint64_t strt, uint64_t stp, int nst, timer_e nme) : start(strt), stop(stp), nest_level(nst), timer_name(nme) {}
    inline uint64_t  getStart() const { return start; }
    inline uint64_t  getStop() const  { return stop;  }
    inline int       getNestLevel() const { return nest_level; }
    inline timer_e   getTimerName() const { return timer_name; }
};

/* ****************************************************************
    Class to implement a dynamically expandable array of events

    ---------------------------------------------------------
    | event 1 | event 2 | event 3 | event 4 | ... | event N |
    ---------------------------------------------------------

    An event is pushed onto the back of this array at every
    explicitTimer->stop() call.  The event records the thread #,
    start time, stop time, and nest level related to the bar width.

    The event vector starts at size INIT_SIZE and grows (doubles in size)
    if needed.  An implication of this behavior is that log(N)
    reallocations are needed (where N is number of events).  If you want
    to avoid reallocations, then set INIT_SIZE to a large value.

    the interface to this class is through six operations:
    1) reset() -- sets the internal_size back to 0 but does not deallocate any memory
    2) size()  -- returns the number of valid elements in the vector
    3) push_back(start, stop, nest, timer_name) -- pushes an event onto
                                                   the back of the array
    4) deallocate() -- frees all memory associated with the vector
    5) sort() -- sorts the vector by start time
    6) operator[index] or at(index) -- returns event reference at that index

**************************************************************** */
class kmp_stats_event_vector {
    kmp_stats_event* events;
    int internal_size;
    int allocated_size;
    static const int INIT_SIZE = 1024;
 public:
    kmp_stats_event_vector() {
        events = (kmp_stats_event*)__kmp_allocate(sizeof(kmp_stats_event)*INIT_SIZE);
        internal_size = 0;
        allocated_size = INIT_SIZE;
    }
   ~kmp_stats_event_vector() {}
    inline void reset() { internal_size = 0; }
    inline int  size() const { return internal_size; }
    void push_back(uint64_t start_time, uint64_t stop_time, int nest_level, timer_e name) {
        int i;
        if(internal_size == allocated_size) {
            kmp_stats_event* tmp = (kmp_stats_event*)__kmp_allocate(sizeof(kmp_stats_event)*allocated_size*2);
            for(i=0;i<internal_size;i++) tmp[i] = events[i];
            __kmp_free(events);
            events = tmp;
            allocated_size*=2;
        }
        events[internal_size] = kmp_stats_event(start_time, stop_time, nest_level, name);
        internal_size++;
        return;
    }
    void deallocate();
    void sort();
    const kmp_stats_event & operator[](int index) const { return events[index]; }
          kmp_stats_event & operator[](int index) { return events[index]; }
    const kmp_stats_event & at(int index) const { return events[index]; }
          kmp_stats_event & at(int index) { return events[index]; }
};

/* ****************************************************************
    Class to implement a doubly-linked, circular, statistics list

    |---| ---> |---| ---> |---| ---> |---| ---> ... next
    |   |      |   |      |   |      |   |
    |---| <--- |---| <--- |---| <--- |---| <--- ... prev
    Sentinel   first      second     third
    Node       node       node       node

    The Sentinel Node is the user handle on the list.
    The first node corresponds to thread 0's statistics.
    The second node corresponds to thread 1's statistics and so on...

    Each node has a _timers, _counters, and _explicitTimers array to
    hold that thread's statistics.  The _explicitTimers
    point to the correct _timer and update its statistics at every stop() call.
    The explicitTimers' pointers are set up in the constructor.
    Each node also has an event vector to hold that thread's timing events.
    The event vector expands as necessary and records the start-stop times
    for each timer.

    The nestLevel variable is for plotting events and is related
    to the bar width in the timeline graph.

    Every thread will have a __thread local pointer to its node in
    the list.  The sentinel node is used by the master thread to
    store "dummy" statistics before __kmp_create_worker() is called.

**************************************************************** */
class kmp_stats_list {
    int gtid;
    timeStat      _timers[TIMER_LAST+1];
    counter       _counters[COUNTER_LAST+1];
    explicitTimer _explicitTimers[EXPLICIT_TIMER_LAST+1];
    partitionedTimers _partitionedTimers;
    int           _nestLevel; // one per thread
    kmp_stats_event_vector _event_vector;
    kmp_stats_list* next;
    kmp_stats_list* prev;
    stats_state_e state;
    int thread_is_idle_flag;
 public:
    kmp_stats_list() : _nestLevel(0), _event_vector(), next(this), prev(this),
      state(IDLE), thread_is_idle_flag(0) {
#define doInit(name,ignore1,ignore2) \
        getExplicitTimer(EXPLICIT_TIMER_##name)->setStat(getTimer(TIMER_##name)); \
        _partitionedTimers.add_timer(EXPLICIT_TIMER_##name, getExplicitTimer(EXPLICIT_TIMER_##name));
        KMP_FOREACH_EXPLICIT_TIMER(doInit,0);
#undef doInit
    }
   ~kmp_stats_list() { }
    inline timeStat *      getTimer(timer_e idx)                  { return &_timers[idx]; }
    inline counter  *      getCounter(counter_e idx)              { return &_counters[idx]; }
    inline explicitTimer * getExplicitTimer(explicit_timer_e idx) { return &_explicitTimers[idx]; }
    inline partitionedTimers * getPartitionedTimers()             { return &_partitionedTimers; }
    inline timeStat *      getTimers()                            { return _timers; }
    inline counter  *      getCounters()                          { return _counters; }
    inline explicitTimer * getExplicitTimers()                    { return _explicitTimers; }
    inline kmp_stats_event_vector & getEventVector()              { return _event_vector; }
    inline void resetEventVector()                                { _event_vector.reset(); }
    inline void incrementNestValue()                              { _nestLevel++; }
    inline int  getNestValue()                                    { return _nestLevel; }
    inline void decrementNestValue()                              { _nestLevel--; }
    inline int  getGtid() const                                   { return gtid; }
    inline void setGtid(int newgtid)                              { gtid = newgtid; }
    inline void setState(stats_state_e newstate)                  { state = newstate; }
    inline stats_state_e getState() const                         { return state; }
    inline stats_state_e * getStatePointer()                      { return &state; }
    inline bool  isIdle()                                         { return thread_is_idle_flag==1; }
    inline void setIdleFlag()                                     { thread_is_idle_flag = 1; }
    inline void resetIdleFlag()                                   { thread_is_idle_flag = 0; }
    kmp_stats_list* push_back(int gtid); // returns newly created list node
    inline void     push_event(uint64_t start_time, uint64_t stop_time, int nest_level, timer_e name) {
        _event_vector.push_back(start_time, stop_time, nest_level, name);
    }
    void deallocate();
    class iterator;
    kmp_stats_list::iterator begin();
    kmp_stats_list::iterator end();
    int size();
    class iterator {
        kmp_stats_list* ptr;
        friend kmp_stats_list::iterator kmp_stats_list::begin();
        friend kmp_stats_list::iterator kmp_stats_list::end();
      public:
        iterator();
       ~iterator();
        iterator operator++();
        iterator operator++(int dummy);
        iterator operator--();
        iterator operator--(int dummy);
        bool operator!=(const iterator & rhs);
        bool operator==(const iterator & rhs);
        kmp_stats_list* operator*() const; // dereference operator
    };
};

/* ****************************************************************
   Class to encapsulate all output functions and the environment variables

   This module holds filenames for various outputs (normal stats, events, plot file),
   as well as coloring information for the plot file.

   The filenames and flags variables are read from environment variables.
   These are read once by the constructor of the global variable __kmp_stats_output
   which calls init().

   During this init() call, event flags for the timeStat::timerInfo[] global array
   are cleared if KMP_STATS_EVENTS is not true (on, 1, yes).

   The only interface function that is public is outputStats(heading).  This function
   should print out everything it needs to, either to files or stderr,
   depending on the environment variables described below

   ENVIRONMENT VARIABLES:
   KMP_STATS_FILE -- if set, all statistics (not events) will be printed to this file,
                     otherwise, print to stderr
   KMP_STATS_THREADS -- if set to "on", then will print per thread statistics to either
                        KMP_STATS_FILE or stderr
   KMP_STATS_PLOT_FILE -- if set, print the ploticus plot file to this filename,
                          otherwise, the plot file is sent to "events.plt"
   KMP_STATS_EVENTS -- if set to "on", then log events, otherwise, don't log events
   KMP_STATS_EVENTS_FILE -- if set, all events are outputted to this file,
                            otherwise, output is sent to "events.dat"

**************************************************************** */
class kmp_stats_output_module {

 public:
    struct rgb_color {
        float r;
        float g;
        float b;
    };

 private:
    std::string outputFileName;
    static const char* eventsFileName;
    static const char* plotFileName;
    static int printPerThreadFlag;
    static int printPerThreadEventsFlag;
    static const rgb_color globalColorArray[];
    static       rgb_color timerColorInfo[];

    void init();
    static void setupEventColors();
    static void printPloticusFile();
    static void printHeaderInfo(FILE *statsOut);
    static void printTimerStats(FILE *statsOut, statistic const * theStats, statistic const * totalStats);
    static void printCounterStats(FILE *statsOut, statistic const * theStats);
    static void printCounters(FILE * statsOut, counter const * theCounters);
    static void printEvents(FILE * eventsOut, kmp_stats_event_vector* theEvents, int gtid);
    static rgb_color getEventColor(timer_e e) { return timerColorInfo[e]; }
    static void windupExplicitTimers();
    bool eventPrintingEnabled() const         { return printPerThreadEventsFlag; }

 public:
    kmp_stats_output_module() { init(); }
    void outputStats(const char* heading);
};

#ifdef __cplusplus
extern "C" {
#endif
void __kmp_stats_init();
void __kmp_reset_stats();
void __kmp_output_stats(const char *);
void __kmp_accumulate_stats_at_exit(void);
// thread local pointer to stats node within list
extern __thread kmp_stats_list* __kmp_stats_thread_ptr;
// head to stats list.
extern kmp_stats_list __kmp_stats_list;
// lock for __kmp_stats_list
extern kmp_tas_lock_t  __kmp_stats_lock;
// reference start time
extern tsc_tick_count __kmp_stats_start_time;
// interface to output
extern kmp_stats_output_module __kmp_stats_output;

#ifdef __cplusplus
}
#endif

// Simple, standard interfaces that drop out completely if stats aren't enabled


/*!
 * \brief Uses specified timer (name) to time code block.
 *
 * @param name timer name as specified under the KMP_FOREACH_TIMER() macro
 *
 * \details Use KMP_TIME_BLOCK(name) macro to time a code block.  This will record the time taken in the block
 * and use the destructor to stop the timer.  Convenient!
 * With this definition you can't have more than one KMP_TIME_BLOCK in the same code block.
 * I don't think that's a problem.
 *
 * @ingroup STATS_GATHERING
*/
#define KMP_TIME_BLOCK(name) \
    blockTimer __BLOCKTIME__(__kmp_stats_thread_ptr->getTimer(TIMER_##name), TIMER_##name)

/*!
 * \brief Adds value to specified timer (name).
 *
 * @param name timer name as specified under the KMP_FOREACH_TIMER() macro
 * @param value double precision sample value to add to statistics for the timer
 *
 * \details Use KMP_COUNT_VALUE(name, value) macro to add a particular value to a timer statistics.
 *
 * @ingroup STATS_GATHERING
*/
#define KMP_COUNT_VALUE(name, value) \
    __kmp_stats_thread_ptr->getTimer(TIMER_##name)->addSample(value)

/*!
 * \brief Increments specified counter (name).
 *
 * @param name counter name as specified under the KMP_FOREACH_COUNTER() macro
 *
 * \details Use KMP_COUNT_BLOCK(name, value) macro to increment a statistics counter for the executing thread.
 *
 * @ingroup STATS_GATHERING
*/
#define KMP_COUNT_BLOCK(name) \
   __kmp_stats_thread_ptr->getCounter(COUNTER_##name)->increment()

/*!
 * \brief "Starts" an explicit timer which will need a corresponding KMP_STOP_EXPLICIT_TIMER() macro.
 *
 * @param name explicit timer name as specified under the KMP_FOREACH_EXPLICIT_TIMER() macro
 *
 * \details Use to start a timer.  This will need a corresponding KMP_STOP_EXPLICIT_TIMER()
 * macro to stop the timer unlike the KMP_TIME_BLOCK(name) macro which has an implicit stopping macro at the end
 * of the code block.  All explicit timers are stopped at library exit time before the final statistics are outputted.
 *
 * @ingroup STATS_GATHERING
*/
#define KMP_START_EXPLICIT_TIMER(name) \
    __kmp_stats_thread_ptr->getExplicitTimer(EXPLICIT_TIMER_##name)->start(TIMER_##name)

/*!
 * \brief "Stops" an explicit timer.
 *
 * @param name explicit timer name as specified under the KMP_FOREACH_EXPLICIT_TIMER() macro
 *
 * \details Use KMP_STOP_EXPLICIT_TIMER(name) to stop a timer.  When this is done, the time between the last KMP_START_EXPLICIT_TIMER(name)
 * and this KMP_STOP_EXPLICIT_TIMER(name) will be added to the timer's stat value.  The timer will then be reset.
 * After the KMP_STOP_EXPLICIT_TIMER(name) macro is called, another call to KMP_START_EXPLICIT_TIMER(name) will start the timer once again.
 *
 * @ingroup STATS_GATHERING
*/
#define KMP_STOP_EXPLICIT_TIMER(name) \
    __kmp_stats_thread_ptr->getExplicitTimer(EXPLICIT_TIMER_##name)->stop(TIMER_##name)

/*!
 * \brief Outputs the current thread statistics and reset them.
 *
 * @param heading_string heading put above the final stats output
 *
 * \details Explicitly stops all timers and outputs all stats.
 * Environment variable, `OMPTB_STATSFILE=filename`, can be used to output the stats to a filename instead of stderr
 * Environment variable, `OMPTB_STATSTHREADS=true|undefined`, can be used to output thread specific stats
 * For now the `OMPTB_STATSTHREADS` environment variable can either be defined with any value, which will print out thread
 * specific stats, or it can be undefined (not specified in the environment) and thread specific stats won't be printed
 * It should be noted that all statistics are reset when this macro is called.
 *
 * @ingroup STATS_GATHERING
*/
#define KMP_OUTPUT_STATS(heading_string) \
    __kmp_output_stats(heading_string)

/*!
 * \brief Initializes the paritioned timers to begin with name.
 *
 * @param name timer which you want this thread to begin with
 *
 * @ingroup STATS_GATHERING
*/
#define KMP_INIT_PARTITIONED_TIMERS(name) \
    __kmp_stats_thread_ptr->getPartitionedTimers()->init(timerPair(EXPLICIT_TIMER_##name, TIMER_##name))

#define KMP_TIME_PARTITIONED_BLOCK(name) \
    blockPartitionedTimer __PBLOCKTIME__(__kmp_stats_thread_ptr->getPartitionedTimers(), \
        timerPair(EXPLICIT_TIMER_##name, TIMER_##name))

#define KMP_PUSH_PARTITIONED_TIMER(name) \
    __kmp_stats_thread_ptr->getPartitionedTimers()->push(timerPair(EXPLICIT_TIMER_##name, TIMER_##name))

#define KMP_POP_PARTITIONED_TIMER() \
    __kmp_stats_thread_ptr->getPartitionedTimers()->pop()

#define KMP_SET_THREAD_STATE(state_name) \
    __kmp_stats_thread_ptr->setState(state_name)

#define KMP_GET_THREAD_STATE() \
    __kmp_stats_thread_ptr->getState()

#define KMP_SET_THREAD_STATE_BLOCK(state_name) \
    blockThreadState __BTHREADSTATE__(__kmp_stats_thread_ptr->getStatePointer(), state_name)

/*!
 * \brief resets all stats (counters to 0, timers to 0 elapsed ticks)
 *
 * \details Reset all stats for all threads.
 *
 * @ingroup STATS_GATHERING
*/
#define KMP_RESET_STATS()  __kmp_reset_stats()

#if (KMP_DEVELOPER_STATS)
# define KMP_TIME_DEVELOPER_BLOCK(n)             KMP_TIME_BLOCK(n)
# define KMP_COUNT_DEVELOPER_VALUE(n,v)          KMP_COUNT_VALUE(n,v)
# define KMP_COUNT_DEVELOPER_BLOCK(n)            KMP_COUNT_BLOCK(n)
# define KMP_START_DEVELOPER_EXPLICIT_TIMER(n)   KMP_START_EXPLICIT_TIMER(n)
# define KMP_STOP_DEVELOPER_EXPLICIT_TIMER(n)    KMP_STOP_EXPLICIT_TIMER(n)
#else
// Null definitions
# define KMP_TIME_DEVELOPER_BLOCK(n)             ((void)0)
# define KMP_COUNT_DEVELOPER_VALUE(n,v)          ((void)0)
# define KMP_COUNT_DEVELOPER_BLOCK(n)            ((void)0)
# define KMP_START_DEVELOPER_EXPLICIT_TIMER(n)   ((void)0)
# define KMP_STOP_DEVELOPER_EXPLICIT_TIMER(n)    ((void)0)
#endif

#else // KMP_STATS_ENABLED

// Null definitions
#define KMP_TIME_BLOCK(n)             ((void)0)
#define KMP_COUNT_VALUE(n,v)          ((void)0)
#define KMP_COUNT_BLOCK(n)            ((void)0)
#define KMP_START_EXPLICIT_TIMER(n)   ((void)0)
#define KMP_STOP_EXPLICIT_TIMER(n)    ((void)0)

#define KMP_OUTPUT_STATS(heading_string) ((void)0)
#define KMP_RESET_STATS()  ((void)0)

#define KMP_TIME_DEVELOPER_BLOCK(n)             ((void)0)
#define KMP_COUNT_DEVELOPER_VALUE(n,v)          ((void)0)
#define KMP_COUNT_DEVELOPER_BLOCK(n)            ((void)0)
#define KMP_START_DEVELOPER_EXPLICIT_TIMER(n)   ((void)0)
#define KMP_STOP_DEVELOPER_EXPLICIT_TIMER(n)    ((void)0)
#define KMP_INIT_PARTITIONED_TIMERS(name)       ((void)0)
#define KMP_TIME_PARTITIONED_BLOCK(name)        ((void)0)
#define KMP_PUSH_PARTITIONED_TIMER(name)        ((void)0)
#define KMP_POP_PARTITIONED_TIMER()             ((void)0)
#define KMP_SET_THREAD_STATE(state_name)        ((void)0)
#define KMP_GET_THREAD_STATE()                  ((void)0)
#define KMP_SET_THREAD_STATE_BLOCK(state_name)  ((void)0)
#endif  // KMP_STATS_ENABLED

#endif // KMP_STATS_H
