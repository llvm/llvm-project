//===---- omptarget.h - OpenMP GPU initialization ---------------- CUDA -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of all library macros, types,
// and functions.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_H
#define OMPTARGET_H

#include "common/allocator.h"
#include "common/debug.h" // debug
#include "common/state-queue.h"
#include "common/support.h"
#include "common/ompd-specific.h"
#include "target_impl.h"
#include "interface.h" // interfaces with omp, compiler, and user

#define OMPTARGET_NVPTX_VERSION 1.1

// used by the library for the interface with the app
#define DISPATCH_FINISHED 0
#define DISPATCH_NOTFINISHED 1

// used by dynamic scheduling
#define FINISHED 0
#define NOT_FINISHED 1
#define LAST_CHUNK 2

#define BARRIER_COUNTER 0
#define ORDERED_COUNTER 1

// Worker slot type which is initialized with the default worker slot
// size of 4*32 bytes.
struct __kmpc_data_sharing_slot {
  __kmpc_data_sharing_slot *Next;
  __kmpc_data_sharing_slot *Prev;
  void *PrevSlotStackPtr;
  void *DataEnd;
  char Data[DS_Worker_Warp_Slot_Size];
};

////////////////////////////////////////////////////////////////////////////////
// task ICV and (implicit & explicit) task state

class omptarget_nvptx_TaskDescr {
#if OMPD_SUPPORT
  friend void __device__ ompd_init( void );
  friend INLINE void ompd_init_thread(
      omptarget_nvptx_TaskDescr *currTaskDescr, void *task_func,
      uint8_t implicit);
  friend __device__ void  ompd_set_device_specific_thread_state(
      omptarget_nvptx_TaskDescr *taskDescr, omp_state_t state);
#endif /* OMPD_SUPPORT */
public:
  // methods for flags
  INLINE omp_sched_t GetRuntimeSched() const;
  INLINE void SetRuntimeSched(omp_sched_t sched);
  INLINE int InParallelRegion() const { return items.flags & TaskDescr_InPar; }
  INLINE int InL2OrHigherParallelRegion() const {
    return items.flags & TaskDescr_InParL2P;
  }
  INLINE int IsParallelConstruct() const {
    return items.flags & TaskDescr_IsParConstr;
  }
  INLINE int IsTaskConstruct() const { return !IsParallelConstruct(); }
  // methods for other fields
  INLINE uint16_t &ThreadId() { return items.threadId; }
  INLINE uint8_t &ParLev() { return items.parLev; }
  INLINE uint64_t &RuntimeChunkSize() { return items.runtimeChunkSize; }
  INLINE omptarget_nvptx_TaskDescr *GetPrevTaskDescr() const { return prev; }
  INLINE void SetPrevTaskDescr(omptarget_nvptx_TaskDescr *taskDescr) {
    prev = taskDescr;
  }
  // init & copy
  INLINE void InitLevelZeroTaskDescr();
  INLINE void InitLevelOneTaskDescr(omptarget_nvptx_TaskDescr *parentTaskDescr);
  INLINE void Copy(omptarget_nvptx_TaskDescr *sourceTaskDescr);
  INLINE void CopyData(omptarget_nvptx_TaskDescr *sourceTaskDescr);
  INLINE void CopyParent(omptarget_nvptx_TaskDescr *parentTaskDescr);
  INLINE void CopyForExplicitTask(omptarget_nvptx_TaskDescr *parentTaskDescr);
  INLINE void CopyToWorkDescr(omptarget_nvptx_TaskDescr *masterTaskDescr);
  INLINE void CopyFromWorkDescr(omptarget_nvptx_TaskDescr *workTaskDescr);
  INLINE void CopyConvergentParent(omptarget_nvptx_TaskDescr *parentTaskDescr,
                                   uint16_t tid, uint16_t tnum);
#ifdef OMPD_SUPPORT
  INLINE ompd_nvptx_thread_info_t *ompd_ThreadInfo() {
    return &ompd_thread_info;
  }
#endif
  INLINE void SaveLoopData();
  INLINE void RestoreLoopData() const;

private:
  // bits for flags: (6 used, 2 free)
  //   3 bits (SchedMask) for runtime schedule
  //   1 bit (InPar) if this thread has encountered one or more parallel region
  //   1 bit (IsParConstr) if ICV for a parallel region (false = explicit task)
  //   1 bit (InParL2+) if this thread has encountered L2 or higher parallel
  //   region
  static const uint8_t TaskDescr_SchedMask = (0x1 | 0x2 | 0x4);
  static const uint8_t TaskDescr_InPar = 0x10;
  static const uint8_t TaskDescr_IsParConstr = 0x20;
  static const uint8_t TaskDescr_InParL2P = 0x40;

  struct SavedLoopDescr_items {
    int64_t loopUpperBound;
    int64_t nextLowerBound;
    int64_t chunk;
    int64_t stride;
    kmp_sched_t schedule;
  } loopData;

  struct TaskDescr_items {
    uint8_t flags; // 6 bit used (see flag above)
    uint8_t parLev;
    uint16_t threadId;         // thread id
    uint64_t runtimeChunkSize; // runtime chunk size
  } items;
#ifdef OMPD_SUPPORT
  ompd_nvptx_thread_info_t ompd_thread_info;
#endif
  omptarget_nvptx_TaskDescr *prev;
};

// build on kmp
typedef struct omptarget_nvptx_ExplicitTaskDescr {
  omptarget_nvptx_TaskDescr
      taskDescr; // omptarget_nvptx task description (must be first)
  kmp_TaskDescr kmpTaskDescr; // kmp task description (must be last)
} omptarget_nvptx_ExplicitTaskDescr;

////////////////////////////////////////////////////////////////////////////////
// Descriptor of a parallel region (worksharing in general)

class omptarget_nvptx_WorkDescr {

public:
  // access to data
  INLINE omptarget_nvptx_TaskDescr *WorkTaskDescr() { return &masterTaskICV; }

private:
  omptarget_nvptx_TaskDescr masterTaskICV;
};

////////////////////////////////////////////////////////////////////////////////

class omptarget_nvptx_TeamDescr {
#ifdef OMPD_SUPPORT
  friend void __device__ ompd_init( void );
#endif /*OMPD_SUPPORT*/
public:
  // access to data
  INLINE omptarget_nvptx_TaskDescr *LevelZeroTaskDescr() {
    return &levelZeroTaskDescr;
  }
  INLINE omptarget_nvptx_WorkDescr &WorkDescr() {
    return workDescrForActiveParallel;
  }

  // init
  INLINE void InitTeamDescr();

  INLINE __kmpc_data_sharing_slot *GetPreallocatedSlotAddr(int wid) {
    worker_rootS[wid].DataEnd =
        &worker_rootS[wid].Data[0] + DS_Worker_Warp_Slot_Size;
    // We currently do not have a next slot.
    worker_rootS[wid].Next = 0;
    worker_rootS[wid].Prev = 0;
    worker_rootS[wid].PrevSlotStackPtr = 0;
    return (__kmpc_data_sharing_slot *)&worker_rootS[wid];
  }

private:
  omptarget_nvptx_TaskDescr
      levelZeroTaskDescr; // icv for team master initial thread
  omptarget_nvptx_WorkDescr
      workDescrForActiveParallel; // one, ONLY for the active par

  ALIGN(16)
  __kmpc_data_sharing_slot worker_rootS[DS_Max_Warp_Number];
};

////////////////////////////////////////////////////////////////////////////////
// thread private data (struct of arrays for better coalescing)
// tid refers here to the global thread id
// do not support multiple concurrent kernel a this time
class omptarget_nvptx_ThreadPrivateContext {
#if OMPD_SUPPORT
  friend void __device__ ompd_init( void );
#endif /* OMPD_SUPPORT */
public:
  // task
  INLINE omptarget_nvptx_TaskDescr *Level1TaskDescr(int tid) {
    return &levelOneTaskDescr[tid];
  }
  INLINE void SetTopLevelTaskDescr(int tid,
                                   omptarget_nvptx_TaskDescr *taskICV) {
    topTaskDescr[tid] = taskICV;
  }
  INLINE omptarget_nvptx_TaskDescr *GetTopLevelTaskDescr(int tid) const;
  // parallel
  INLINE uint16_t &NumThreadsForNextParallel(int tid) {
    return nextRegion.tnum[tid];
  }
  // schedule (for dispatch)
  INLINE kmp_sched_t &ScheduleType(int tid) { return schedule[tid]; }
  INLINE int64_t &Chunk(int tid) { return chunk[tid]; }
  INLINE int64_t &LoopUpperBound(int tid) { return loopUpperBound[tid]; }
  INLINE int64_t &NextLowerBound(int tid) { return nextLowerBound[tid]; }
  INLINE int64_t &Stride(int tid) { return stride[tid]; }

  INLINE omptarget_nvptx_TeamDescr &TeamContext() { return teamContext; }

  INLINE void InitThreadPrivateContext(int tid);
  INLINE uint64_t &Cnt() { return cnt; }

private:
  // team context for this team
  omptarget_nvptx_TeamDescr teamContext;
  // task ICV for implicit threads in the only parallel region
  omptarget_nvptx_TaskDescr levelOneTaskDescr[MAX_THREADS_PER_TEAM];
  // pointer where to find the current task ICV (top of the stack)
  omptarget_nvptx_TaskDescr *topTaskDescr[MAX_THREADS_PER_TEAM];
  union {
    // Only one of the two is live at the same time.
    // parallel
    uint16_t tnum[MAX_THREADS_PER_TEAM];
  } nextRegion;
  // schedule (for dispatch)
  kmp_sched_t schedule[MAX_THREADS_PER_TEAM]; // remember schedule type for #for
  int64_t chunk[MAX_THREADS_PER_TEAM];
  int64_t loopUpperBound[MAX_THREADS_PER_TEAM];
  // state for dispatch with dyn/guided OR static (never use both at a time)
  int64_t nextLowerBound[MAX_THREADS_PER_TEAM];
  int64_t stride[MAX_THREADS_PER_TEAM];
  uint64_t cnt;
#ifdef OMPD_SUPPORT
  // The implicit parallel region around the master task in generic mode
  ompd_nvptx_parallel_info_t ompd_levelZeroParallelInfo;
#endif
};

/// Memory manager for statically allocated memory.
class omptarget_nvptx_SimpleMemoryManager {
private:
  struct MemDataTy {
    volatile unsigned keys[OMP_STATE_COUNT];
  } MemData[MAX_SM] ALIGN(128);

  INLINE static uint32_t hash(unsigned key) {
    return key & (OMP_STATE_COUNT - 1);
  }

public:
  INLINE void Release();
  INLINE const void *Acquire(const void *buf, size_t size);
};

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// global data tables
////////////////////////////////////////////////////////////////////////////////

extern omptarget_nvptx_SimpleMemoryManager omptarget_nvptx_simpleMemoryManager;
extern uint32_t EXTERN_SHARED(usedMemIdx);
extern uint32_t EXTERN_SHARED(usedSlotIdx);
#if _OPENMP
extern uint32_t volatile parallelLevel[MAX_THREADS_PER_TEAM / WARPSIZE];
#pragma omp allocate(parallelLevel) allocator(omp_pteam_mem_alloc)
#else
extern uint8_t EXTERN_SHARED(parallelLevel)[MAX_THREADS_PER_TEAM / WARPSIZE];
#endif
extern uint16_t EXTERN_SHARED(threadLimit);
extern uint16_t EXTERN_SHARED(threadsInTeam);
extern uint16_t EXTERN_SHARED(nThreads);
extern omptarget_nvptx_ThreadPrivateContext *
    EXTERN_SHARED(omptarget_nvptx_threadPrivateContext);

extern int8_t EXTERN_SHARED(execution_param);
extern void *EXTERN_SHARED(ReductionScratchpadPtr);

////////////////////////////////////////////////////////////////////////////////
// work function (outlined parallel/simd functions) and arguments.
// needed for L1 parallelism only.
////////////////////////////////////////////////////////////////////////////////

typedef void *omptarget_nvptx_WorkFn;
extern omptarget_nvptx_WorkFn EXTERN_SHARED(omptarget_nvptx_workFn);

////////////////////////////////////////////////////////////////////////////////
// get private data structures
////////////////////////////////////////////////////////////////////////////////

INLINE omptarget_nvptx_TeamDescr &getMyTeamDescriptor();
INLINE omptarget_nvptx_WorkDescr &getMyWorkDescriptor();
INLINE omptarget_nvptx_TaskDescr *
getMyTopTaskDescriptor(bool isSPMDExecutionMode);
INLINE omptarget_nvptx_TaskDescr *getMyTopTaskDescriptor(int globalThreadId);

////////////////////////////////////////////////////////////////////////////////
// inlined implementation
////////////////////////////////////////////////////////////////////////////////

INLINE uint32_t __kmpc_impl_ffs(uint32_t x) { return __builtin_ffs(x); }
INLINE uint32_t __kmpc_impl_popc(uint32_t x) { return __builtin_popcount(x); }
INLINE uint32_t __kmpc_impl_ffs(uint64_t x) { return __builtin_ffsl(x); }
INLINE uint32_t __kmpc_impl_popc(uint64_t x) { return __builtin_popcountl(x); }

#include "common/omptargeti.h"

#endif
