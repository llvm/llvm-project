//===---- omptarget-nvptx.h - NVPTX OpenMP GPU initialization ---- CUDA -*-===//
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

#ifndef __OMPTARGET_NVPTX_H
#define __OMPTARGET_NVPTX_H

// std includes
#include <stdint.h>
#include <stdlib.h>

#include <inttypes.h>

// cuda includes
#include <cuda.h>
#include <math.h>

// local includes
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
#define __ACTIVEMASK() __activemask()
#else
#define __SHFL_SYNC(mask, var, srcLane) __shfl((var), (srcLane))
#define __SHFL_DOWN_SYNC(mask, var, delta, width)                              \
  __shfl_down((var), (delta), (width))
#define __ACTIVEMASK() __ballot(1)
#endif

#define __SYNCTHREADS_N(n) asm volatile("bar.sync %0;" : : "r"(n) : "memory");
#define __SYNCTHREADS() __SYNCTHREADS_N(0)

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
  INLINE void **GetArgs() const { return args; };
private:
  // buffer of pre-allocated arguments.
  void *buffer[MAX_SHARED_ARGS];
  // pointer to arguments buffer.
  // starts off as a pointer to 'buffer' but can be dynamically allocated.
  void **args;
  // starts off as MAX_SHARED_ARGS but can increase in size.
  uint32_t nArgs;
};

extern __device__ __shared__ omptarget_nvptx_SharedArgs
    omptarget_nvptx_globalArgs;

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
  // The size of the preallocated shared memory buffer per team
  DS_Shared_Memory_Size = 128,
};

// Data structure to keep in shared memory that traces the current slot, stack,
// and frame pointer as well as the active threads that didn't exit the current
// environment.
struct DataSharingStateTy {
  __kmpc_data_sharing_slot *SlotPtr[DS_Max_Warp_Number];
  void *StackPtr[DS_Max_Warp_Number];
  void * volatile FramePtr[DS_Max_Warp_Number];
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
  INLINE uint16_t &NThreads() { return items.nthreads; }
  INLINE uint16_t &ThreadLimit() { return items.threadlimit; }
  INLINE uint16_t &ThreadId() { return items.threadId; }
  INLINE uint16_t &ThreadsInTeam() { return items.threadsInTeam; }
  INLINE uint64_t &RuntimeChunkSize() { return items.runtimeChunkSize; }
  INLINE omptarget_nvptx_TaskDescr *GetPrevTaskDescr() const { return prev; }
  INLINE void SetPrevTaskDescr(omptarget_nvptx_TaskDescr *taskDescr) {
    prev = taskDescr;
  }
  // init & copy
  INLINE void InitLevelZeroTaskDescr(bool isSPMDExecutionMode);
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
  INLINE omptarget_nvptx_TaskDescr *WorkTaskDescr() { return &masterTaskICV; }

private:
  omptarget_nvptx_TaskDescr masterTaskICV;
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
  INLINE uint64_t *getLastprivateIterBuffer() { return &lastprivateIterBuffer; }

  // init
  INLINE void InitTeamDescr(bool isSPMDExecutionMode);

  INLINE __kmpc_data_sharing_slot *RootS(int wid, bool IsMasterThread) {
    // If this is invoked by the master thread of the master warp then intialize
    // it with a smaller slot.
    if (IsMasterThread) {
      // Do not initalize this slot again if it has already been initalized.
      if (master_rootS[0].DataEnd == &master_rootS[0].Data[0] + DS_Slot_Size)
        return 0;
      // Initialize the pointer to the end of the slot given the size of the
      // data section. DataEnd is non-inclusive.
      master_rootS[0].DataEnd = &master_rootS[0].Data[0] + DS_Slot_Size;
      // We currently do not have a next slot.
      master_rootS[0].Next = 0;
      master_rootS[0].Prev = 0;
      master_rootS[0].PrevSlotStackPtr = 0;
      return (__kmpc_data_sharing_slot *)&master_rootS[0];
    }
    // Do not initalize this slot again if it has already been initalized.
    if (worker_rootS[wid].DataEnd ==
        &worker_rootS[wid].Data[0] + DS_Worker_Warp_Slot_Size)
      return 0;
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
  uint64_t lastprivateIterBuffer;

  __align__(16)
      __kmpc_data_sharing_worker_slot_static worker_rootS[WARPSIZE];
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
  INLINE omptarget_nvptx_TaskDescr *GetTopLevelTaskDescr(int tid) const;
  // parallel
  INLINE uint16_t &NumThreadsForNextParallel(int tid) {
    return nextRegion.tnum[tid];
  }
  // simd
  INLINE uint16_t &SimdLimitForNextSimd(int tid) {
    return nextRegion.slim[tid];
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
  // schedule (for dispatch)
  kmp_sched_t schedule[MAX_THREADS_PER_TEAM]; // remember schedule type for #for
  int64_t chunk[MAX_THREADS_PER_TEAM];
  int64_t loopUpperBound[MAX_THREADS_PER_TEAM];
  // state for dispatch with dyn/guided OR static (never use both at a time)
  int64_t nextLowerBound[MAX_THREADS_PER_TEAM];
  int64_t stride[MAX_THREADS_PER_TEAM];
  uint64_t cnt;
};

/// Device envrionment data
struct omptarget_device_environmentTy {
  int32_t debug_level;
};

/// Memory manager for statically allocated memory.
class omptarget_nvptx_SimpleMemoryManager {
private:
  __align__(128) struct MemDataTy {
    volatile unsigned keys[OMP_STATE_COUNT];
  } MemData[MAX_SM];

  INLINE static uint32_t hash(unsigned key) {
    return key & (OMP_STATE_COUNT - 1);
  }

public:
  INLINE void Release();
  INLINE const void *Acquire(const void *buf, size_t size);
};

////////////////////////////////////////////////////////////////////////////////
// global device envrionment
////////////////////////////////////////////////////////////////////////////////

extern __device__ omptarget_device_environmentTy omptarget_device_environment;

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// global data tables
////////////////////////////////////////////////////////////////////////////////

extern __device__ omptarget_nvptx_SimpleMemoryManager
    omptarget_nvptx_simpleMemoryManager;
extern __device__ __shared__ uint32_t usedMemIdx;
extern __device__ __shared__ uint32_t usedSlotIdx;
extern __device__ __shared__ uint8_t parallelLevel;
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
INLINE omptarget_nvptx_TaskDescr *
getMyTopTaskDescriptor(bool isSPMDExecutionMode);
INLINE omptarget_nvptx_TaskDescr *getMyTopTaskDescriptor(int globalThreadId);

////////////////////////////////////////////////////////////////////////////////
// inlined implementation
////////////////////////////////////////////////////////////////////////////////

#include "omptarget-nvptxi.h"
#include "supporti.h"

#endif
