//===------- interface.h - NVPTX OpenMP interface definitions ---- CUDA -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains debug macros to be used in the application.
//
//  This file contains all the definitions that are relevant to
//  the interface. The first section contains the interface as
//  declared by OpenMP.  A second section includes library private calls
//  (mostly debug, temporary?) The third section includes the compiler
//  specific interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef _INTERFACES_H_
#define _INTERFACES_H_

////////////////////////////////////////////////////////////////////////////////
// OpenMP interface
////////////////////////////////////////////////////////////////////////////////

typedef uint32_t omp_lock_t;      /* arbitrary type of the right length */
typedef uint64_t omp_nest_lock_t; /* arbitrary type of the right length */

typedef enum omp_sched_t {
  omp_sched_static = 1,  /* chunkSize >0 */
  omp_sched_dynamic = 2, /* chunkSize >0 */
  omp_sched_guided = 3,  /* chunkSize >0 */
  omp_sched_auto = 4,    /* no chunkSize */
} omp_sched_t;

typedef enum omp_proc_bind_t {
  omp_proc_bind_false = 0,
  omp_proc_bind_true = 1,
  omp_proc_bind_master = 2,
  omp_proc_bind_close = 3,
  omp_proc_bind_spread = 4
} omp_proc_bind_t;

EXTERN double omp_get_wtick(void);
EXTERN double omp_get_wtime(void);

EXTERN void omp_set_num_threads(int num);
EXTERN int omp_get_num_threads(void);
EXTERN int omp_get_max_threads(void);
EXTERN int omp_get_thread_limit(void);
EXTERN int omp_get_thread_num(void);
EXTERN int omp_get_num_procs(void);
EXTERN int omp_in_parallel(void);
EXTERN int omp_in_final(void);
EXTERN void omp_set_dynamic(int flag);
EXTERN int omp_get_dynamic(void);
EXTERN void omp_set_nested(int flag);
EXTERN int omp_get_nested(void);
EXTERN void omp_set_max_active_levels(int level);
EXTERN int omp_get_max_active_levels(void);
EXTERN int omp_get_level(void);
EXTERN int omp_get_active_level(void);
EXTERN int omp_get_ancestor_thread_num(int level);
EXTERN int omp_get_team_size(int level);

EXTERN void omp_init_lock(omp_lock_t *lock);
EXTERN void omp_init_nest_lock(omp_nest_lock_t *lock);
EXTERN void omp_destroy_lock(omp_lock_t *lock);
EXTERN void omp_destroy_nest_lock(omp_nest_lock_t *lock);
EXTERN void omp_set_lock(omp_lock_t *lock);
EXTERN void omp_set_nest_lock(omp_nest_lock_t *lock);
EXTERN void omp_unset_lock(omp_lock_t *lock);
EXTERN void omp_unset_nest_lock(omp_nest_lock_t *lock);
EXTERN int omp_test_lock(omp_lock_t *lock);
EXTERN int omp_test_nest_lock(omp_nest_lock_t *lock);

EXTERN void omp_get_schedule(omp_sched_t *kind, int *modifier);
EXTERN void omp_set_schedule(omp_sched_t kind, int modifier);
EXTERN omp_proc_bind_t omp_get_proc_bind(void);
EXTERN int omp_get_cancellation(void);
EXTERN void omp_set_default_device(int deviceId);
EXTERN int omp_get_default_device(void);
EXTERN int omp_get_num_devices(void);
EXTERN int omp_get_num_teams(void);
EXTERN int omp_get_team_num(void);
EXTERN int omp_is_initial_device(void);
EXTERN int omp_get_initial_device(void);
EXTERN int omp_get_max_task_priority(void);

////////////////////////////////////////////////////////////////////////////////
// OMPTARGET_NVPTX private (debug / temportary?) interface
////////////////////////////////////////////////////////////////////////////////

// for debug
EXTERN void __kmpc_print_str(char *title);
EXTERN void __kmpc_print_title_int(char *title, int data);
EXTERN void __kmpc_print_index(char *title, int i);
EXTERN void __kmpc_print_int(int data);
EXTERN void __kmpc_print_double(double data);
EXTERN void __kmpc_print_address_int64(int64_t data);

////////////////////////////////////////////////////////////////////////////////
// file below is swiped from kmpc host interface
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// kmp specifc types
////////////////////////////////////////////////////////////////////////////////

typedef enum kmp_sched_t {
  kmp_sched_static_chunk = 33,
  kmp_sched_static_nochunk = 34,
  kmp_sched_dynamic = 35,
  kmp_sched_guided = 36,
  kmp_sched_runtime = 37,
  kmp_sched_auto = 38,

  kmp_sched_static_ordered = 65,
  kmp_sched_static_nochunk_ordered = 66,
  kmp_sched_dynamic_ordered = 67,
  kmp_sched_guided_ordered = 68,
  kmp_sched_runtime_ordered = 69,
  kmp_sched_auto_ordered = 70,

  kmp_sched_distr_static_chunk = 91,
  kmp_sched_distr_static_nochunk = 92,
  kmp_sched_distr_static_chunk_sched_static_chunkone = 93,

  kmp_sched_default = kmp_sched_static_nochunk,
  kmp_sched_unordered_first = kmp_sched_static_chunk,
  kmp_sched_unordered_last = kmp_sched_auto,
  kmp_sched_ordered_first = kmp_sched_static_ordered,
  kmp_sched_ordered_last = kmp_sched_auto_ordered,
  kmp_sched_distribute_first = kmp_sched_distr_static_chunk,
  kmp_sched_distribute_last =
      kmp_sched_distr_static_chunk_sched_static_chunkone,

  /* Support for OpenMP 4.5 monotonic and nonmonotonic schedule modifiers.
   * Since we need to distinguish the three possible cases (no modifier,
   * monotonic modifier, nonmonotonic modifier), we need separate bits for
   * each modifier. The absence of monotonic does not imply nonmonotonic,
   * especially since 4.5 says that the behaviour of the "no modifier" case
   * is implementation defined in 4.5, but will become "nonmonotonic" in 5.0.
   *
   * Since we're passing a full 32 bit value, we can use a couple of high
   * bits for these flags; out of paranoia we avoid the sign bit.
   *
   * These modifiers can be or-ed into non-static schedules by the compiler
   * to pass the additional information. They will be stripped early in the
   * processing in __kmp_dispatch_init when setting up schedules, so
   * most of the code won't ever see schedules with these bits set.
   */
  kmp_sched_modifier_monotonic = (1 << 29),
  /**< Set if the monotonic schedule modifier was present */
  kmp_sched_modifier_nonmonotonic = (1 << 30),
/**< Set if the nonmonotonic schedule modifier was present */

#define SCHEDULE_WITHOUT_MODIFIERS(s)                                          \
  (enum kmp_sched_t)(                                                          \
      (s) & ~(kmp_sched_modifier_nonmonotonic | kmp_sched_modifier_monotonic))
#define SCHEDULE_HAS_MONOTONIC(s) (((s)&kmp_sched_modifier_monotonic) != 0)
#define SCHEDULE_HAS_NONMONOTONIC(s)                                           \
  (((s)&kmp_sched_modifier_nonmonotonic) != 0)
#define SCHEDULE_HAS_NO_MODIFIERS(s)                                           \
  (((s) & (kmp_sched_modifier_nonmonotonic | kmp_sched_modifier_monotonic)) == \
   0)

} kmp_sched_t;

// parallel defs
typedef void kmp_Indent;
typedef void (*kmp_ParFctPtr)(int32_t *global_tid, int32_t *bound_tid, ...);
typedef void (*kmp_ReductFctPtr)(void *lhsData, void *rhsData);
typedef void (*kmp_InterWarpCopyFctPtr)(void *src, int32_t warp_num);
typedef void (*kmp_ShuffleReductFctPtr)(void *rhsData, int16_t lane_id,
                                        int16_t lane_offset,
                                        int16_t shortCircuit);
typedef void (*kmp_CopyToScratchpadFctPtr)(void *reduceData, void *scratchpad,
                                           int32_t index, int32_t width);
typedef void (*kmp_LoadReduceFctPtr)(void *reduceData, void *scratchpad,
                                     int32_t index, int32_t width,
                                     int32_t reduce);

// task defs
typedef struct kmp_TaskDescr kmp_TaskDescr;
typedef int32_t (*kmp_TaskFctPtr)(int32_t global_tid, kmp_TaskDescr *taskDescr);
typedef struct kmp_TaskDescr {
  void *sharedPointerTable;   // ptr to a table of shared var ptrs
  kmp_TaskFctPtr sub;         // task subroutine
  int32_t partId;             // unused
  kmp_TaskFctPtr destructors; // destructor of c++ first private
} kmp_TaskDescr;
// task dep defs
#define KMP_TASKDEP_IN 0x1u
#define KMP_TASKDEP_OUT 0x2u
typedef struct kmp_TaskDep_Public {
  void *addr;
  size_t len;
  uint8_t flags; // bit 0: in, bit 1: out
} kmp_TaskDep_Public;

// flags that interpret the interface part of tasking flags
#define KMP_TASK_IS_TIED 0x1
#define KMP_TASK_FINAL 0x2
#define KMP_TASK_MERGED_IF0 0x4 /* unused */
#define KMP_TASK_DESTRUCTOR_THUNK 0x8

// flags for task setup return
#define KMP_CURRENT_TASK_NOT_SUSPENDED 0
#define KMP_CURRENT_TASK_SUSPENDED 1

// sync defs
typedef int32_t kmp_CriticalName[8];

////////////////////////////////////////////////////////////////////////////////
// flags for kstate (all bits initially off)
////////////////////////////////////////////////////////////////////////////////

// first 2 bits used by kmp_Reduction (defined in kmp_reduction.cpp)
#define KMP_REDUCTION_MASK 0x3
#define KMP_SKIP_NEXT_CALL 0x4
#define KMP_SKIP_NEXT_CANCEL_BARRIER 0x8

////////////////////////////////////////////////////////////////////////////////
// data
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// external interface
////////////////////////////////////////////////////////////////////////////////

// query
EXTERN int32_t __kmpc_global_num_threads(kmp_Indent *loc); // missing
EXTERN int32_t __kmpc_bound_thread_num(kmp_Indent *loc);   // missing
EXTERN int32_t __kmpc_bound_num_threads(kmp_Indent *loc);  // missing
EXTERN int32_t __kmpc_in_parallel(kmp_Indent *loc);        // missing

// parallel
EXTERN int32_t __kmpc_global_thread_num(kmp_Indent *loc);
EXTERN void __kmpc_push_num_threads(kmp_Indent *loc, int32_t global_tid,
                                    int32_t num_threads);
// simd
EXTERN void __kmpc_push_simd_limit(kmp_Indent *loc, int32_t global_tid,
                                   int32_t simd_limit);
// aee ... not supported
// EXTERN void __kmpc_fork_call(kmp_Indent *loc, int32_t argc, kmp_ParFctPtr
// microtask, ...);
EXTERN void __kmpc_serialized_parallel(kmp_Indent *loc, uint32_t global_tid);
EXTERN void __kmpc_end_serialized_parallel(kmp_Indent *loc,
                                           uint32_t global_tid);
EXTERN uint16_t __kmpc_parallel_level(kmp_Indent *loc, uint32_t global_tid);

// proc bind
EXTERN void __kmpc_push_proc_bind(kmp_Indent *loc, uint32_t global_tid,
                                  int proc_bind);
EXTERN int omp_get_num_places(void);
EXTERN int omp_get_place_num_procs(int place_num);
EXTERN void omp_get_place_proc_ids(int place_num, int *ids);
EXTERN int omp_get_place_num(void);
EXTERN int omp_get_partition_num_places(void);
EXTERN void omp_get_partition_place_nums(int *place_nums);

// for static (no chunk or chunk)
EXTERN void __kmpc_for_static_init_4(kmp_Indent *loc, int32_t global_tid,
                                     int32_t sched, int32_t *plastiter,
                                     int32_t *plower, int32_t *pupper,
                                     int32_t *pstride, int32_t incr,
                                     int32_t chunk);
EXTERN void __kmpc_for_static_init_4u(kmp_Indent *loc, int32_t global_tid,
                                      int32_t sched, int32_t *plastiter,
                                      uint32_t *plower, uint32_t *pupper,
                                      int32_t *pstride, int32_t incr,
                                      int32_t chunk);
EXTERN void __kmpc_for_static_init_8(kmp_Indent *loc, int32_t global_tid,
                                     int32_t sched, int32_t *plastiter,
                                     int64_t *plower, int64_t *pupper,
                                     int64_t *pstride, int64_t incr,
                                     int64_t chunk);
EXTERN void __kmpc_for_static_init_8u(kmp_Indent *loc, int32_t global_tid,
                                      int32_t sched, int32_t *plastiter1,
                                      uint64_t *plower, uint64_t *pupper,
                                      int64_t *pstride, int64_t incr,
                                      int64_t chunk);
EXTERN
void __kmpc_for_static_init_4_simple_spmd(kmp_Indent *loc, int32_t global_tid,
                                          int32_t sched, int32_t *plastiter,
                                          int32_t *plower, int32_t *pupper,
                                          int32_t *pstride, int32_t incr,
                                          int32_t chunk);
EXTERN
void __kmpc_for_static_init_4u_simple_spmd(kmp_Indent *loc, int32_t global_tid,
                                           int32_t sched, int32_t *plastiter,
                                           uint32_t *plower, uint32_t *pupper,
                                           int32_t *pstride, int32_t incr,
                                           int32_t chunk);
EXTERN
void __kmpc_for_static_init_8_simple_spmd(kmp_Indent *loc, int32_t global_tid,
                                          int32_t sched, int32_t *plastiter,
                                          int64_t *plower, int64_t *pupper,
                                          int64_t *pstride, int64_t incr,
                                          int64_t chunk);
EXTERN
void __kmpc_for_static_init_8u_simple_spmd(kmp_Indent *loc, int32_t global_tid,
                                           int32_t sched, int32_t *plastiter1,
                                           uint64_t *plower, uint64_t *pupper,
                                           int64_t *pstride, int64_t incr,
                                           int64_t chunk);
EXTERN
void __kmpc_for_static_init_4_simple_generic(kmp_Indent *loc,
                                             int32_t global_tid, int32_t sched,
                                             int32_t *plastiter,
                                             int32_t *plower, int32_t *pupper,
                                             int32_t *pstride, int32_t incr,
                                             int32_t chunk);
EXTERN
void __kmpc_for_static_init_4u_simple_generic(
    kmp_Indent *loc, int32_t global_tid, int32_t sched, int32_t *plastiter,
    uint32_t *plower, uint32_t *pupper, int32_t *pstride, int32_t incr,
    int32_t chunk);
EXTERN
void __kmpc_for_static_init_8_simple_generic(kmp_Indent *loc,
                                             int32_t global_tid, int32_t sched,
                                             int32_t *plastiter,
                                             int64_t *plower, int64_t *pupper,
                                             int64_t *pstride, int64_t incr,
                                             int64_t chunk);
EXTERN
void __kmpc_for_static_init_8u_simple_generic(
    kmp_Indent *loc, int32_t global_tid, int32_t sched, int32_t *plastiter1,
    uint64_t *plower, uint64_t *pupper, int64_t *pstride, int64_t incr,
    int64_t chunk);

EXTERN void __kmpc_for_static_fini(kmp_Indent *loc, int32_t global_tid);

// for dynamic
EXTERN void __kmpc_dispatch_init_4(kmp_Indent *loc, int32_t global_tid,
                                   int32_t sched, int32_t lower, int32_t upper,
                                   int32_t incr, int32_t chunk);
EXTERN void __kmpc_dispatch_init_4u(kmp_Indent *loc, int32_t global_tid,
                                    int32_t sched, uint32_t lower,
                                    uint32_t upper, int32_t incr,
                                    int32_t chunk);
EXTERN void __kmpc_dispatch_init_8(kmp_Indent *loc, int32_t global_tid,
                                   int32_t sched, int64_t lower, int64_t upper,
                                   int64_t incr, int64_t chunk);
EXTERN void __kmpc_dispatch_init_8u(kmp_Indent *loc, int32_t global_tid,
                                    int32_t sched, uint64_t lower,
                                    uint64_t upper, int64_t incr,
                                    int64_t chunk);

EXTERN int __kmpc_dispatch_next_4(kmp_Indent *loc, int32_t global_tid,
                                  int32_t *plastiter, int32_t *plower,
                                  int32_t *pupper, int32_t *pstride);
EXTERN int __kmpc_dispatch_next_4u(kmp_Indent *loc, int32_t global_tid,
                                   int32_t *plastiter, uint32_t *plower,
                                   uint32_t *pupper, int32_t *pstride);
EXTERN int __kmpc_dispatch_next_8(kmp_Indent *loc, int32_t global_tid,
                                  int32_t *plastiter, int64_t *plower,
                                  int64_t *pupper, int64_t *pstride);
EXTERN int __kmpc_dispatch_next_8u(kmp_Indent *loc, int32_t global_tid,
                                   int32_t *plastiter, uint64_t *plower,
                                   uint64_t *pupper, int64_t *pstride);

EXTERN void __kmpc_dispatch_fini_4(kmp_Indent *loc, int32_t global_tid);
EXTERN void __kmpc_dispatch_fini_4u(kmp_Indent *loc, int32_t global_tid);
EXTERN void __kmpc_dispatch_fini_8(kmp_Indent *loc, int32_t global_tid);
EXTERN void __kmpc_dispatch_fini_8u(kmp_Indent *loc, int32_t global_tid);

// Support for reducing conditional lastprivate variables
EXTERN void __kmpc_reduce_conditional_lastprivate(kmp_Indent *loc,
                                                  int32_t global_tid,
                                                  int32_t varNum, void *array);

// reduction
EXTERN void __kmpc_nvptx_end_reduce(int32_t global_tid);
EXTERN void __kmpc_nvptx_end_reduce_nowait(int32_t global_tid);
EXTERN int32_t __kmpc_nvptx_parallel_reduce_nowait(
    int32_t global_tid, int32_t num_vars, size_t reduce_size, void *reduce_data,
    kmp_ShuffleReductFctPtr shflFct, kmp_InterWarpCopyFctPtr cpyFct);
EXTERN int32_t __kmpc_nvptx_parallel_reduce_nowait_simple_spmd(
    int32_t global_tid, int32_t num_vars, size_t reduce_size, void *reduce_data,
    kmp_ShuffleReductFctPtr shflFct, kmp_InterWarpCopyFctPtr cpyFct);
EXTERN int32_t __kmpc_nvptx_parallel_reduce_nowait_simple_generic(
    int32_t global_tid, int32_t num_vars, size_t reduce_size, void *reduce_data,
    kmp_ShuffleReductFctPtr shflFct, kmp_InterWarpCopyFctPtr cpyFct);
EXTERN int32_t __kmpc_nvptx_simd_reduce_nowait(
    int32_t global_tid, int32_t num_vars, size_t reduce_size, void *reduce_data,
    kmp_ShuffleReductFctPtr shflFct, kmp_InterWarpCopyFctPtr cpyFct);
EXTERN int32_t __kmpc_nvptx_teams_reduce_nowait(
    int32_t global_tid, int32_t num_vars, size_t reduce_size, void *reduce_data,
    kmp_ShuffleReductFctPtr shflFct, kmp_InterWarpCopyFctPtr cpyFct,
    kmp_CopyToScratchpadFctPtr sratchFct, kmp_LoadReduceFctPtr ldFct);
EXTERN int32_t __kmpc_nvptx_teams_reduce_nowait_simple_spmd(
    int32_t global_tid, int32_t num_vars, size_t reduce_size, void *reduce_data,
    kmp_ShuffleReductFctPtr shflFct, kmp_InterWarpCopyFctPtr cpyFct,
    kmp_CopyToScratchpadFctPtr sratchFct, kmp_LoadReduceFctPtr ldFct);
EXTERN int32_t __kmpc_nvptx_teams_reduce_nowait_simple_generic(
    int32_t global_tid, int32_t num_vars, size_t reduce_size, void *reduce_data,
    kmp_ShuffleReductFctPtr shflFct, kmp_InterWarpCopyFctPtr cpyFct,
    kmp_CopyToScratchpadFctPtr sratchFct, kmp_LoadReduceFctPtr ldFct);
EXTERN int32_t __kmpc_shuffle_int32(int32_t val, int16_t delta, int16_t size);
EXTERN int64_t __kmpc_shuffle_int64(int64_t val, int16_t delta, int16_t size);

// sync barrier
EXTERN void __kmpc_barrier(kmp_Indent *loc_ref, int32_t tid);
EXTERN void __kmpc_barrier_simple_spmd(kmp_Indent *loc_ref, int32_t tid);
EXTERN void __kmpc_barrier_simple_generic(kmp_Indent *loc_ref, int32_t tid);
EXTERN int32_t __kmpc_cancel_barrier(kmp_Indent *loc, int32_t global_tid);

// single
EXTERN int32_t __kmpc_single(kmp_Indent *loc, int32_t global_tid);
EXTERN void __kmpc_end_single(kmp_Indent *loc, int32_t global_tid);

// sync
EXTERN int32_t __kmpc_master(kmp_Indent *loc, int32_t global_tid);
EXTERN void __kmpc_end_master(kmp_Indent *loc, int32_t global_tid);
EXTERN void __kmpc_ordered(kmp_Indent *loc, int32_t global_tid);
EXTERN void __kmpc_end_ordered(kmp_Indent *loc, int32_t global_tid);
EXTERN void __kmpc_critical(kmp_Indent *loc, int32_t global_tid,
                            kmp_CriticalName *crit);
EXTERN void __kmpc_end_critical(kmp_Indent *loc, int32_t global_tid,
                                kmp_CriticalName *crit);
EXTERN void __kmpc_flush(kmp_Indent *loc);

// vote
EXTERN int32_t __kmpc_warp_active_thread_mask();

// tasks
EXTERN kmp_TaskDescr *__kmpc_omp_task_alloc(kmp_Indent *loc,
                                            uint32_t global_tid, int32_t flag,
                                            size_t sizeOfTaskInclPrivate,
                                            size_t sizeOfSharedTable,
                                            kmp_TaskFctPtr sub);
EXTERN int32_t __kmpc_omp_task(kmp_Indent *loc, uint32_t global_tid,
                               kmp_TaskDescr *newLegacyTaskDescr);
EXTERN int32_t __kmpc_omp_task_with_deps(kmp_Indent *loc, uint32_t global_tid,
                                         kmp_TaskDescr *newLegacyTaskDescr,
                                         int32_t depNum, void *depList,
                                         int32_t noAliasDepNum,
                                         void *noAliasDepList);
EXTERN void __kmpc_omp_task_begin_if0(kmp_Indent *loc, uint32_t global_tid,
                                      kmp_TaskDescr *newLegacyTaskDescr);
EXTERN void __kmpc_omp_task_complete_if0(kmp_Indent *loc, uint32_t global_tid,
                                         kmp_TaskDescr *newLegacyTaskDescr);
EXTERN void __kmpc_omp_wait_deps(kmp_Indent *loc, uint32_t global_tid,
                                 int32_t depNum, void *depList,
                                 int32_t noAliasDepNum, void *noAliasDepList);
EXTERN void __kmpc_taskgroup(kmp_Indent *loc, uint32_t global_tid);
EXTERN void __kmpc_end_taskgroup(kmp_Indent *loc, uint32_t global_tid);
EXTERN int32_t __kmpc_omp_taskyield(kmp_Indent *loc, uint32_t global_tid,
                                    int end_part);
EXTERN int32_t __kmpc_omp_taskwait(kmp_Indent *loc, uint32_t global_tid);
EXTERN void __kmpc_taskloop(kmp_Indent *loc, uint32_t global_tid,
                            kmp_TaskDescr *newKmpTaskDescr, int if_val,
                            uint64_t *lb, uint64_t *ub, int64_t st, int nogroup,
                            int32_t sched, uint64_t grainsize, void *task_dup);

// cancel
EXTERN int32_t __kmpc_cancellationpoint(kmp_Indent *loc, int32_t global_tid,
                                        int32_t cancelVal);
EXTERN int32_t __kmpc_cancel(kmp_Indent *loc, int32_t global_tid,
                             int32_t cancelVal);

// non standard
EXTERN void __kmpc_kernel_init_params(void *ReductionScratchpadPtr);
EXTERN void __kmpc_kernel_init(int ThreadLimit, int16_t RequiresOMPRuntime);
EXTERN void __kmpc_kernel_deinit(int16_t IsOMPRuntimeInitialized);
EXTERN void __kmpc_spmd_kernel_init(int ThreadLimit, int16_t RequiresOMPRuntime,
                                    int16_t RequiresDataSharing);
EXTERN void __kmpc_spmd_kernel_deinit();
EXTERN void __kmpc_kernel_prepare_parallel(void *WorkFn,
                                           int16_t IsOMPRuntimeInitialized);
EXTERN bool __kmpc_kernel_parallel(void **WorkFn,
                                   int16_t IsOMPRuntimeInitialized);
EXTERN void __kmpc_kernel_end_parallel();
EXTERN bool __kmpc_kernel_convergent_parallel(void *buffer, uint32_t Mask,
                                              bool *IsFinal,
                                              int32_t *LaneSource);
EXTERN void __kmpc_kernel_end_convergent_parallel(void *buffer);
EXTERN bool __kmpc_kernel_convergent_simd(void *buffer, uint32_t Mask,
                                          bool *IsFinal, int32_t *LaneSource,
                                          int32_t *LaneId, int32_t *NumLanes);
EXTERN void __kmpc_kernel_end_convergent_simd(void *buffer);


EXTERN void __kmpc_data_sharing_init_stack();
EXTERN void *__kmpc_data_sharing_push_stack(size_t size, int16_t UseSharedMemory);
EXTERN void __kmpc_data_sharing_pop_stack(void *a);
EXTERN void __kmpc_begin_sharing_variables(void ***GlobalArgs, size_t nArgs);
EXTERN void __kmpc_end_sharing_variables();
EXTERN void __kmpc_get_shared_variables(void ***GlobalArgs);

// The slot used for data sharing by the master and worker threads. We use a
// complete (default size version and an incomplete one so that we allow sizes
// greater than the default).
struct __kmpc_data_sharing_slot {
  __kmpc_data_sharing_slot *Next;
  __kmpc_data_sharing_slot *Prev;
  void *PrevSlotStackPtr;
  void *DataEnd;
  char Data[];
};
EXTERN void
__kmpc_initialize_data_sharing_environment(__kmpc_data_sharing_slot *RootS,
                                           size_t InitialDataSize);
EXTERN void *__kmpc_data_sharing_environment_begin(
    __kmpc_data_sharing_slot **SavedSharedSlot, void **SavedSharedStack,
    void **SavedSharedFrame, int32_t *SavedActiveThreads,
    size_t SharingDataSize, size_t SharingDefaultDataSize,
    int16_t IsOMPRuntimeInitialized);
EXTERN void __kmpc_data_sharing_environment_end(
    __kmpc_data_sharing_slot **SavedSharedSlot, void **SavedSharedStack,
    void **SavedSharedFrame, int32_t *SavedActiveThreads, int32_t IsEntryPoint);

EXTERN void *
__kmpc_get_data_sharing_environment_frame(int32_t SourceThreadID,
                                          int16_t IsOMPRuntimeInitialized);
#endif
