//===---- omptarget-nvptx.h - NVPTX OpenMP GPU initialization ---- CUDA -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of all library macros, types,
// and functions.
//
//===----------------------------------------------------------------------===//

#ifndef __OMPTARGET_NVPTX_H
#define __OMPTARGET_NVPTX_H

// std includes
#include <stdint.h>
#include <stdlib.h>

// cuda includes
#include <cuda.h>
#include <math.h>

// local includes
#include "counter_group.h"
#include "debug.h"     // debug
#include "interface.h" // interfaces with omp, compiler, and user
#include "option.h"    // choices we have
#include "state-queue.h"
#include "support.h"

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

// Macros for Cuda intrinsics
// In Cuda 9.0, the *_sync() version takes an extra argument 'mask'.
// Also, __ballot(1) in Cuda 8.0 is replaced with __activemask().
#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000
#define __SHFL_SYNC(mask, var, srcLane) __shfl_sync((mask), (var), (srcLane))
#define __SHFL_DOWN_SYNC(mask, var, delta, width)                              \
  __shfl_down_sync((mask), (var), (delta), (width))
#define __BALLOT_SYNC(mask, predicate) __ballot_sync((mask), (predicate))
#define __ACTIVEMASK() __activemask()
#else
#define __SHFL_SYNC(mask, var, srcLane) __shfl((var), (srcLane))
#define __SHFL_DOWN_SYNC(mask, var, delta, width)                              \
  __shfl_down((var), (delta), (width))
#define __BALLOT_SYNC(mask, predicate) __ballot((predicate))
#define __ACTIVEMASK() __ballot(1)
#endif

// arguments needed for L0 parallelism only.
class omptarget_nvptx_SharedArgs {
public:
  // All these methods must be called by the master thread only.
  INLINE void Init() {
    args  = buffer;
    nArgs = MAX_SHARED_ARGS;
  }
  INLINE void DeInit() {
    // Free any memory allocated for outlined parallel function with a large
    // number of arguments.
    if (nArgs > MAX_SHARED_ARGS) {
      SafeFree(args, (char *)"new extended args");
      Init();
    }
  }
  INLINE void EnsureSize(size_t size) {
    if (size > nArgs) {
      if (nArgs > MAX_SHARED_ARGS) {
        SafeFree(args, (char *)"new extended args");
      }
      args = (void **) SafeMalloc(size * sizeof(void *),
                                  (char *)"new extended args");
      nArgs = size;
    }
  }
  // Called by all threads.
  INLINE void **GetArgs() { return args; };
private:
  // buffer of pre-allocated arguments.
  void *buffer[MAX_SHARED_ARGS];
  // pointer to arguments buffer.
  // starts off as a pointer to 'buffer' but can be dynamically allocated.
  void **args;
  // starts off as MAX_SHARED_ARGS but can increase in size.
  uint32_t nArgs;
};

extern __device__ __shared__ omptarget_nvptx_SharedArgs omptarget_nvptx_globalArgs;

// Data sharing related quantities, need to match what is used in the compiler.
enum DATA_SHARING_SIZES {
  // The maximum number of workers in a kernel.
  DS_Max_Worker_Threads = 992,
  // The size reserved for data in a shared memory slot.
  DS_Slot_Size = 256,
  // The slot size that should be reserved for a working warp.
  DS_Worker_Warp_Slot_Size = WARPSIZE * DS_Slot_Size,
  // The maximum number of warps in use
  DS_Max_Warp_Number = 32,
};

// Data structure to keep in shared memory that traces the current slot, stack,
// and frame pointer as well as the active threads that didn't exit the current
// environment.
struct DataSharingStateTy {
  __kmpc_data_sharing_slot *SlotPtr[DS_Max_Warp_Number];
  void *StackPtr[DS_Max_Warp_Number];
  __kmpc_data_sharing_slot *TailPtr[DS_Max_Warp_Number];
  void *FramePtr[DS_Max_Warp_Number];
  int32_t ActiveThreads[DS_Max_Warp_Number];
};
// Additional worker slot type which is initialized with the default worker slot
// size of 4*32 bytes.
struct __kmpc_data_sharing_worker_slot_static {
  __kmpc_data_sharing_slot *Next;
  __kmpc_data_sharing_slot *Prev;
  void *PrevSlotStackPtr;
  void *DataEnd;
  char Data[DS_Worker_Warp_Slot_Size];
};
// Additional master slot type which is initialized with the default master slot
// size of 4 bytes.
struct __kmpc_data_sharing_master_slot_static {
  __kmpc_data_sharing_slot *Next;
  __kmpc_data_sharing_slot *Prev;
  void *PrevSlotStackPtr;
  void *DataEnd;
  char Data[DS_Slot_Size];
};
extern __device__ __shared__ DataSharingStateTy DataSharingState;

////////////////////////////////////////////////////////////////////////////////
// task ICV and (implicit & explicit) task state

class omptarget_nvptx_TaskDescr {
public:
  // methods for flags
  INLINE omp_sched_t GetRuntimeSched();
  INLINE void SetRuntimeSched(omp_sched_t sched);
  INLINE int IsDynamic() { return items.flags & TaskDescr_IsDynamic; }
  INLINE void SetDynamic() {
    items.flags = items.flags | TaskDescr_IsDynamic;
  }
  INLINE void ClearDynamic() {
    items.flags = items.flags & (~TaskDescr_IsDynamic);
  }
  INLINE int InParallelRegion() { return items.flags & TaskDescr_InPar; }
  INLINE int InL2OrHigherParallelRegion() {
    return items.flags & TaskDescr_InParL2P;
  }
  INLINE int IsParallelConstruct() {
    return items.flags & TaskDescr_IsParConstr;
  }
  INLINE int IsTaskConstruct() { return !IsParallelConstruct(); }
  // methods for other fields
  INLINE uint16_t &NThreads() { return items.nthreads; }
  INLINE uint16_t &ThreadLimit() { return items.threadlimit; }
  INLINE uint16_t &ThreadId() { return items.threadId; }
  INLINE uint16_t &ThreadsInTeam() { return items.threadsInTeam; }
  INLINE uint64_t &RuntimeChunkSize() { return items.runtimeChunkSize; }
  INLINE omptarget_nvptx_TaskDescr *GetPrevTaskDescr() { return prev; }
  INLINE void SetPrevTaskDescr(omptarget_nvptx_TaskDescr *taskDescr) {
    prev = taskDescr;
  }
  // init & copy
  INLINE void InitLevelZeroTaskDescr();
  INLINE void InitLevelOneTaskDescr(uint16_t tnum,
                                    omptarget_nvptx_TaskDescr *parentTaskDescr);
  INLINE void Copy(omptarget_nvptx_TaskDescr *sourceTaskDescr);
  INLINE void CopyData(omptarget_nvptx_TaskDescr *sourceTaskDescr);
  INLINE void CopyParent(omptarget_nvptx_TaskDescr *parentTaskDescr);
  INLINE void CopyForExplicitTask(omptarget_nvptx_TaskDescr *parentTaskDescr);
  INLINE void CopyToWorkDescr(omptarget_nvptx_TaskDescr *masterTaskDescr,
                              uint16_t tnum);
  INLINE void CopyFromWorkDescr(omptarget_nvptx_TaskDescr *workTaskDescr);
  INLINE void CopyConvergentParent(omptarget_nvptx_TaskDescr *parentTaskDescr,
                                   uint16_t tid, uint16_t tnum);

private:
  // bits for flags: (7 used, 1 free)
  //   3 bits (SchedMask) for runtime schedule
  //   1 bit (IsDynamic) for dynamic schedule (false = static)
  //   1 bit (InPar) if this thread has encountered one or more parallel region
  //   1 bit (IsParConstr) if ICV for a parallel region (false = explicit task)
  //   1 bit (InParL2+) if this thread has encountered L2 or higher parallel
  //   region
  static const uint8_t TaskDescr_SchedMask = (0x1 | 0x2 | 0x4);
  static const uint8_t TaskDescr_IsDynamic = 0x8;
  static const uint8_t TaskDescr_InPar = 0x10;
  static const uint8_t TaskDescr_IsParConstr = 0x20;
  static const uint8_t TaskDescr_InParL2P = 0x40;

  struct TaskDescr_items {
    uint8_t flags; // 6 bit used (see flag above)
    uint8_t unused;
    uint16_t nthreads;         // thread num for subsequent parallel regions
    uint16_t threadlimit;      // thread limit ICV
    uint16_t threadId;         // thread id
    uint16_t threadsInTeam;    // threads in current team
    uint64_t runtimeChunkSize; // runtime chunk size
  } items;
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
  INLINE omptarget_nvptx_CounterGroup &CounterGroup() { return cg; }
  INLINE omptarget_nvptx_TaskDescr *WorkTaskDescr() { return &masterTaskICV; }
  // init
  INLINE void InitWorkDescr();

private:
  omptarget_nvptx_CounterGroup cg; // for barrier (no other needed)
  omptarget_nvptx_TaskDescr masterTaskICV;
  bool hasCancel;
};

////////////////////////////////////////////////////////////////////////////////

class omptarget_nvptx_TeamDescr {
public:
  // access to data
  INLINE omptarget_nvptx_TaskDescr *LevelZeroTaskDescr() {
    return &levelZeroTaskDescr;
  }
  INLINE omptarget_nvptx_WorkDescr &WorkDescr() {
    return workDescrForActiveParallel;
  }
  INLINE omp_lock_t *CriticalLock() { return &criticalLock; }
  INLINE uint64_t *getLastprivateIterBuffer() { return &lastprivateIterBuffer; }

  // init
  INLINE void InitTeamDescr();

  INLINE __kmpc_data_sharing_slot *RootS(int wid) {
    // If this is invoked by the master thread of the master warp then intialize
    // it with a smaller slot.
    if (wid == WARPSIZE - 1) {
      // Initialize the pointer to the end of the slot given the size of the
      // data section. DataEnd is non-inclusive.
      master_rootS[0].DataEnd = &master_rootS[0].Data[0] + DS_Slot_Size;
      // We currently do not have a next slot.
      master_rootS[0].Next = 0;
      master_rootS[0].Prev = 0;
      master_rootS[0].PrevSlotStackPtr = 0;
      return (__kmpc_data_sharing_slot *)&master_rootS[0];
    }
    // Initialize the pointer to the end of the slot given the size of the data
    // section. DataEnd is non-inclusive.
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
  omp_lock_t criticalLock;
  uint64_t lastprivateIterBuffer;

  __align__(16)
      __kmpc_data_sharing_worker_slot_static worker_rootS[WARPSIZE - 1];
  __align__(16) __kmpc_data_sharing_master_slot_static master_rootS[1];
};

////////////////////////////////////////////////////////////////////////////////
// thread private data (struct of arrays for better coalescing)
// tid refers here to the global thread id
// do not support multiple concurrent kernel a this time
class omptarget_nvptx_ThreadPrivateContext {
public:
  // task
  INLINE omptarget_nvptx_TaskDescr *Level1TaskDescr(int tid) {
    return &levelOneTaskDescr[tid];
  }
  INLINE void SetTopLevelTaskDescr(int tid,
                                   omptarget_nvptx_TaskDescr *taskICV) {
    topTaskDescr[tid] = taskICV;
  }
  INLINE omptarget_nvptx_TaskDescr *GetTopLevelTaskDescr(int tid);
  // parallel
  INLINE uint16_t &NumThreadsForNextParallel(int tid) {
    return nextRegion.tnum[tid];
  }
  // simd
  INLINE uint16_t &SimdLimitForNextSimd(int tid) {
    return nextRegion.slim[tid];
  }
  // sync
  INLINE Counter &Priv(int tid) { return priv[tid]; }
  INLINE void IncrementPriv(int tid, Counter val) { priv[tid] += val; }
  // schedule (for dispatch)
  INLINE kmp_sched_t &ScheduleType(int tid) { return schedule[tid]; }
  INLINE int64_t &Chunk(int tid) { return chunk[tid]; }
  INLINE int64_t &LoopUpperBound(int tid) { return loopUpperBound[tid]; }
  // state for dispatch with dyn/guided
  INLINE Counter &CurrentEvent(int tid) {
    return currEvent_or_nextLowerBound[tid];
  }
  INLINE Counter &EventsNumber(int tid) { return eventsNum_or_stride[tid]; }
  // state for dispatch with static
  INLINE Counter &NextLowerBound(int tid) {
    return currEvent_or_nextLowerBound[tid];
  }
  INLINE Counter &Stride(int tid) { return eventsNum_or_stride[tid]; }

  INLINE omptarget_nvptx_TeamDescr &TeamContext() { return teamContext; }

  INLINE void InitThreadPrivateContext(int tid);
  INLINE void SetSourceQueue(uint64_t Src) { SourceQueue = Src; }
  INLINE uint64_t GetSourceQueue() { return SourceQueue; }

private:
  // team context for this team
  omptarget_nvptx_TeamDescr teamContext;
  // task ICV for implict threads in the only parallel region
  omptarget_nvptx_TaskDescr levelOneTaskDescr[MAX_THREADS_PER_TEAM];
  // pointer where to find the current task ICV (top of the stack)
  omptarget_nvptx_TaskDescr *topTaskDescr[MAX_THREADS_PER_TEAM];
  union {
    // Only one of the two is live at the same time.
    // parallel
    uint16_t tnum[MAX_THREADS_PER_TEAM];
    // simd limit
    uint16_t slim[MAX_THREADS_PER_TEAM];
  } nextRegion;
  // sync
  Counter priv[MAX_THREADS_PER_TEAM];
  // schedule (for dispatch)
  kmp_sched_t schedule[MAX_THREADS_PER_TEAM]; // remember schedule type for #for
  int64_t chunk[MAX_THREADS_PER_TEAM];
  int64_t loopUpperBound[MAX_THREADS_PER_TEAM];
  // state for dispatch with dyn/guided OR static (never use both at a time)
  Counter currEvent_or_nextLowerBound[MAX_THREADS_PER_TEAM];
  Counter eventsNum_or_stride[MAX_THREADS_PER_TEAM];
  // Queue to which this object must be returned.
  uint64_t SourceQueue;
};

////////////////////////////////////////////////////////////////////////////////
// global data tables
////////////////////////////////////////////////////////////////////////////////

extern __device__ __shared__
    omptarget_nvptx_ThreadPrivateContext *omptarget_nvptx_threadPrivateContext;
extern __device__ __shared__ uint32_t execution_param;
extern __device__ __shared__ void *ReductionScratchpadPtr;

////////////////////////////////////////////////////////////////////////////////
// work function (outlined parallel/simd functions) and arguments.
// needed for L1 parallelism only.
////////////////////////////////////////////////////////////////////////////////

typedef void *omptarget_nvptx_WorkFn;
extern volatile __device__ __shared__ omptarget_nvptx_WorkFn
    omptarget_nvptx_workFn;

////////////////////////////////////////////////////////////////////////////////
// get private data structures
////////////////////////////////////////////////////////////////////////////////

INLINE omptarget_nvptx_TeamDescr &getMyTeamDescriptor();
INLINE omptarget_nvptx_WorkDescr &getMyWorkDescriptor();
INLINE omptarget_nvptx_TaskDescr *getMyTopTaskDescriptor();
INLINE omptarget_nvptx_TaskDescr *getMyTopTaskDescriptor(int globalThreadId);

////////////////////////////////////////////////////////////////////////////////
// inlined implementation
////////////////////////////////////////////////////////////////////////////////

#include "counter_groupi.h"
#include "omptarget-nvptxi.h"
#include "supporti.h"

#endif
