//===----- Workshare.cpp -  OpenMP workshare implementation ------ C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the KMPC interface
// for the loop construct plus other worksharing constructs that use the same
// interface as loops.
//
//===----------------------------------------------------------------------===//

#include "Debug.h"
#include "Interface.h"
#include "Mapping.h"
#include "State.h"
#include "Synchronization.h"
#include "Types.h"
#include "Utils.h"

using namespace _OMP;

// TODO:
struct DynamicScheduleTracker {
  int64_t Chunk;
  int64_t LoopUpperBound;
  int64_t NextLowerBound;
  int64_t Stride;
  kmp_sched_t ScheduleType;
  DynamicScheduleTracker *NextDST;
};

#define ASSERT0(...)

// used by the library for the interface with the app
#define DISPATCH_FINISHED 0
#define DISPATCH_NOTFINISHED 1

// used by dynamic scheduling
#define FINISHED 0
#define NOT_FINISHED 1
#define LAST_CHUNK 2

#pragma omp declare target

// TODO: This variable is a hack inherited from the old runtime.
uint64_t SHARED(Cnt);

template <typename T, typename ST> struct omptarget_nvptx_LoopSupport {
  ////////////////////////////////////////////////////////////////////////////////
  // Loop with static scheduling with chunk

  // Generic implementation of OMP loop scheduling with static policy
  /*! \brief Calculate initial bounds for static loop and stride
   *  @param[in] loc location in code of the call (not used here)
   *  @param[in] global_tid global thread id
   *  @param[in] schetype type of scheduling (see omptarget-nvptx.h)
   *  @param[in] plastiter pointer to last iteration
   *  @param[in,out] pointer to loop lower bound. it will contain value of
   *  lower bound of first chunk
   *  @param[in,out] pointer to loop upper bound. It will contain value of
   *  upper bound of first chunk
   *  @param[in,out] pointer to loop stride. It will contain value of stride
   *  between two successive chunks executed by the same thread
   *  @param[in] loop increment bump
   *  @param[in] chunk size
   */

  // helper function for static chunk
  static void ForStaticChunk(int &last, T &lb, T &ub, ST &stride, ST chunk,
                             T entityId, T numberOfEntities) {
    // each thread executes multiple chunks all of the same size, except
    // the last one
    // distance between two successive chunks
    stride = numberOfEntities * chunk;
    lb = lb + entityId * chunk;
    T inputUb = ub;
    ub = lb + chunk - 1; // Clang uses i <= ub
    // Say ub' is the begining of the last chunk. Then who ever has a
    // lower bound plus a multiple of the increment equal to ub' is
    // the last one.
    T beginingLastChunk = inputUb - (inputUb % chunk);
    last = ((beginingLastChunk - lb) % stride) == 0;
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Loop with static scheduling without chunk

  // helper function for static no chunk
  static void ForStaticNoChunk(int &last, T &lb, T &ub, ST &stride, ST &chunk,
                               T entityId, T numberOfEntities) {
    // No chunk size specified.  Each thread or warp gets at most one
    // chunk; chunks are all almost of equal size
    T loopSize = ub - lb + 1;

    chunk = loopSize / numberOfEntities;
    T leftOver = loopSize - chunk * numberOfEntities;

    if (entityId < leftOver) {
      chunk++;
      lb = lb + entityId * chunk;
    } else {
      lb = lb + entityId * chunk + leftOver;
    }

    T inputUb = ub;
    ub = lb + chunk - 1; // Clang uses i <= ub
    last = lb <= inputUb && inputUb <= ub;
    stride = loopSize; // make sure we only do 1 chunk per warp
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Support for Static Init

  static void for_static_init(int32_t gtid, int32_t schedtype,
                              int32_t *plastiter, T *plower, T *pupper,
                              ST *pstride, ST chunk, bool IsSPMDExecutionMode) {
    // When IsRuntimeUninitialized is true, we assume that the caller is
    // in an L0 parallel region and that all worker threads participate.

    // Assume we are in teams region or that we use a single block
    // per target region
    int numberOfActiveOMPThreads = omp_get_num_threads();

    // All warps that are in excess of the maximum requested, do
    // not execute the loop
    ASSERT0(LT_FUSSY, gtid < numberOfActiveOMPThreads,
            "current thread is not needed here; error");

    // copy
    int lastiter = 0;
    T lb = *plower;
    T ub = *pupper;
    ST stride = *pstride;

    // init
    switch (SCHEDULE_WITHOUT_MODIFIERS(schedtype)) {
    case kmp_sched_static_chunk: {
      if (chunk > 0) {
        ForStaticChunk(lastiter, lb, ub, stride, chunk, gtid,
                       numberOfActiveOMPThreads);
        break;
      }
    } // note: if chunk <=0, use nochunk
    case kmp_sched_static_balanced_chunk: {
      if (chunk > 0) {
        // round up to make sure the chunk is enough to cover all iterations
        T tripCount = ub - lb + 1; // +1 because ub is inclusive
        T span = (tripCount + numberOfActiveOMPThreads - 1) /
                 numberOfActiveOMPThreads;
        // perform chunk adjustment
        chunk = (span + chunk - 1) & ~(chunk - 1);

        ASSERT0(LT_FUSSY, ub >= lb, "ub must be >= lb.");
        T oldUb = ub;
        ForStaticChunk(lastiter, lb, ub, stride, chunk, gtid,
                       numberOfActiveOMPThreads);
        if (ub > oldUb)
          ub = oldUb;
        break;
      }
    } // note: if chunk <=0, use nochunk
    case kmp_sched_static_nochunk: {
      ForStaticNoChunk(lastiter, lb, ub, stride, chunk, gtid,
                       numberOfActiveOMPThreads);
      break;
    }
    case kmp_sched_distr_static_chunk: {
      if (chunk > 0) {
        ForStaticChunk(lastiter, lb, ub, stride, chunk, omp_get_team_num(),
                       omp_get_num_teams());
        break;
      } // note: if chunk <=0, use nochunk
    }
    case kmp_sched_distr_static_nochunk: {
      ForStaticNoChunk(lastiter, lb, ub, stride, chunk, omp_get_team_num(),
                       omp_get_num_teams());
      break;
    }
    case kmp_sched_distr_static_chunk_sched_static_chunkone: {
      ForStaticChunk(lastiter, lb, ub, stride, chunk,
                     numberOfActiveOMPThreads * omp_get_team_num() + gtid,
                     omp_get_num_teams() * numberOfActiveOMPThreads);
      break;
    }
    default: {
      // ASSERT(LT_FUSSY, 0, "unknown schedtype %d", (int)schedtype);
      ForStaticChunk(lastiter, lb, ub, stride, chunk, gtid,
                     numberOfActiveOMPThreads);
      break;
    }
    }
    // copy back
    *plastiter = lastiter;
    *plower = lb;
    *pupper = ub;
    *pstride = stride;
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Support for dispatch Init

  static int OrderedSchedule(kmp_sched_t schedule) {
    return schedule >= kmp_sched_ordered_first &&
           schedule <= kmp_sched_ordered_last;
  }

  static void dispatch_init(IdentTy *loc, int32_t threadId,
                            kmp_sched_t schedule, T lb, T ub, ST st, ST chunk,
                            DynamicScheduleTracker *DST) {
    int tid = mapping::getThreadIdInBlock();
    T tnum = omp_get_num_threads();
    T tripCount = ub - lb + 1; // +1 because ub is inclusive
    ASSERT0(LT_FUSSY, threadId < tnum,
            "current thread is not needed here; error");

    /* Currently just ignore the monotonic and non-monotonic modifiers
     * (the compiler isn't producing them * yet anyway).
     * When it is we'll want to look at them somewhere here and use that
     * information to add to our schedule choice. We shouldn't need to pass
     * them on, they merely affect which schedule we can legally choose for
     * various dynamic cases. (In particular, whether or not a stealing scheme
     * is legal).
     */
    schedule = SCHEDULE_WITHOUT_MODIFIERS(schedule);

    // Process schedule.
    if (tnum == 1 || tripCount <= 1 || OrderedSchedule(schedule)) {
      if (OrderedSchedule(schedule))
        __kmpc_barrier(loc, threadId);
      schedule = kmp_sched_static_chunk;
      chunk = tripCount; // one thread gets the whole loop
    } else if (schedule == kmp_sched_runtime) {
      // process runtime
      omp_sched_t rtSched;
      int ChunkInt;
      omp_get_schedule(&rtSched, &ChunkInt);
      chunk = ChunkInt;
      switch (rtSched) {
      case omp_sched_static: {
        if (chunk > 0)
          schedule = kmp_sched_static_chunk;
        else
          schedule = kmp_sched_static_nochunk;
        break;
      }
      case omp_sched_auto: {
        schedule = kmp_sched_static_chunk;
        chunk = 1;
        break;
      }
      case omp_sched_dynamic:
      case omp_sched_guided: {
        schedule = kmp_sched_dynamic;
        break;
      }
      }
    } else if (schedule == kmp_sched_auto) {
      schedule = kmp_sched_static_chunk;
      chunk = 1;
    } else {
      // ASSERT(LT_FUSSY,
      //        schedule == kmp_sched_dynamic || schedule == kmp_sched_guided,
      //        "unknown schedule %d & chunk %lld\n", (int)schedule,
      //        (long long)chunk);
    }

    // init schedules
    if (schedule == kmp_sched_static_chunk) {
      ASSERT0(LT_FUSSY, chunk > 0, "bad chunk value");
      // save sched state
      DST->ScheduleType = schedule;
      // save ub
      DST->LoopUpperBound = ub;
      // compute static chunk
      ST stride;
      int lastiter = 0;
      ForStaticChunk(lastiter, lb, ub, stride, chunk, threadId, tnum);
      // save computed params
      DST->Chunk = chunk;
      DST->NextLowerBound = lb;
      DST->Stride = stride;
    } else if (schedule == kmp_sched_static_balanced_chunk) {
      ASSERT0(LT_FUSSY, chunk > 0, "bad chunk value");
      // save sched state
      DST->ScheduleType = schedule;
      // save ub
      DST->LoopUpperBound = ub;
      // compute static chunk
      ST stride;
      int lastiter = 0;
      // round up to make sure the chunk is enough to cover all iterations
      T span = (tripCount + tnum - 1) / tnum;
      // perform chunk adjustment
      chunk = (span + chunk - 1) & ~(chunk - 1);

      T oldUb = ub;
      ForStaticChunk(lastiter, lb, ub, stride, chunk, threadId, tnum);
      ASSERT0(LT_FUSSY, ub >= lb, "ub must be >= lb.");
      if (ub > oldUb)
        ub = oldUb;
      // save computed params
      DST->Chunk = chunk;
      DST->NextLowerBound = lb;
      DST->Stride = stride;
    } else if (schedule == kmp_sched_static_nochunk) {
      ASSERT0(LT_FUSSY, chunk == 0, "bad chunk value");
      // save sched state
      DST->ScheduleType = schedule;
      // save ub
      DST->LoopUpperBound = ub;
      // compute static chunk
      ST stride;
      int lastiter = 0;
      ForStaticNoChunk(lastiter, lb, ub, stride, chunk, threadId, tnum);
      // save computed params
      DST->Chunk = chunk;
      DST->NextLowerBound = lb;
      DST->Stride = stride;
    } else if (schedule == kmp_sched_dynamic || schedule == kmp_sched_guided) {
      // save data
      DST->ScheduleType = schedule;
      if (chunk < 1)
        chunk = 1;
      DST->Chunk = chunk;
      DST->LoopUpperBound = ub;
      DST->NextLowerBound = lb;
      __kmpc_barrier(loc, threadId);
      if (tid == 0) {
        Cnt = 0;
        fence::team(__ATOMIC_SEQ_CST);
      }
      __kmpc_barrier(loc, threadId);
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Support for dispatch next

  static uint64_t NextIter() {
    __kmpc_impl_lanemask_t active = mapping::activemask();
    uint32_t leader = utils::ffs(active) - 1;
    uint32_t change = utils::popc(active);
    __kmpc_impl_lanemask_t lane_mask_lt = mapping::lanemaskLT();
    unsigned int rank = utils::popc(active & lane_mask_lt);
    uint64_t warp_res;
    if (rank == 0) {
      warp_res = atomic::add(&Cnt, change, __ATOMIC_SEQ_CST);
    }
    warp_res = utils::shuffle(active, warp_res, leader);
    return warp_res + rank;
  }

  static int DynamicNextChunk(T &lb, T &ub, T chunkSize, T loopLowerBound,
                              T loopUpperBound) {
    T N = NextIter();
    lb = loopLowerBound + N * chunkSize;
    ub = lb + chunkSize - 1; // Clang uses i <= ub

    // 3 result cases:
    //  a. lb and ub < loopUpperBound --> NOT_FINISHED
    //  b. lb < loopUpperBound and ub >= loopUpperBound: last chunk -->
    //  NOT_FINISHED
    //  c. lb and ub >= loopUpperBound: empty chunk --> FINISHED
    // a.
    if (lb <= loopUpperBound && ub < loopUpperBound) {
      return NOT_FINISHED;
    }
    // b.
    if (lb <= loopUpperBound) {
      ub = loopUpperBound;
      return LAST_CHUNK;
    }
    // c. if we are here, we are in case 'c'
    lb = loopUpperBound + 2;
    ub = loopUpperBound + 1;
    return FINISHED;
  }

  static int dispatch_next(IdentTy *loc, int32_t gtid, int32_t *plast,
                           T *plower, T *pupper, ST *pstride,
                           DynamicScheduleTracker *DST) {
    // ID of a thread in its own warp

    // automatically selects thread or warp ID based on selected implementation
    ASSERT0(LT_FUSSY, gtid < omp_get_num_threads(),
            "current thread is not needed here; error");
    // retrieve schedule
    kmp_sched_t schedule = DST->ScheduleType;

    // xxx reduce to one
    if (schedule == kmp_sched_static_chunk ||
        schedule == kmp_sched_static_nochunk) {
      T myLb = DST->NextLowerBound;
      T ub = DST->LoopUpperBound;
      // finished?
      if (myLb > ub) {
        return DISPATCH_FINISHED;
      }
      // not finished, save current bounds
      ST chunk = DST->Chunk;
      *plower = myLb;
      T myUb = myLb + chunk - 1; // Clang uses i <= ub
      if (myUb > ub)
        myUb = ub;
      *pupper = myUb;
      *plast = (int32_t)(myUb == ub);

      // increment next lower bound by the stride
      ST stride = DST->Stride;
      DST->NextLowerBound = myLb + stride;
      return DISPATCH_NOTFINISHED;
    }
    ASSERT0(LT_FUSSY,
            schedule == kmp_sched_dynamic || schedule == kmp_sched_guided,
            "bad sched");
    T myLb, myUb;
    int finished = DynamicNextChunk(myLb, myUb, DST->Chunk, DST->NextLowerBound,
                                    DST->LoopUpperBound);

    if (finished == FINISHED)
      return DISPATCH_FINISHED;

    // not finished (either not finished or last chunk)
    *plast = (int32_t)(finished == LAST_CHUNK);
    *plower = myLb;
    *pupper = myUb;
    *pstride = 1;

    return DISPATCH_NOTFINISHED;
  }

  static void dispatch_fini() {
    // nothing
  }

  ////////////////////////////////////////////////////////////////////////////////
  // end of template class that encapsulate all the helper functions
  ////////////////////////////////////////////////////////////////////////////////
};

////////////////////////////////////////////////////////////////////////////////
// KMP interface implementation (dyn loops)
////////////////////////////////////////////////////////////////////////////////

// TODO: This is a stopgap. We probably want to expand the dispatch API to take
//       an DST pointer which can then be allocated properly without malloc.
DynamicScheduleTracker *THREAD_LOCAL(ThreadDSTPtr);

// Create a new DST, link the current one, and define the new as current.
static DynamicScheduleTracker *pushDST() {
  DynamicScheduleTracker *NewDST = static_cast<DynamicScheduleTracker *>(
      memory::allocGlobal(sizeof(DynamicScheduleTracker), "new DST"));
  *NewDST = DynamicScheduleTracker({0});
  NewDST->NextDST = ThreadDSTPtr;
  ThreadDSTPtr = NewDST;
  return ThreadDSTPtr;
}

// Return the current DST.
static DynamicScheduleTracker *peekDST() { return ThreadDSTPtr; }

// Pop the current DST and restore the last one.
static void popDST() {
  DynamicScheduleTracker *OldDST = ThreadDSTPtr->NextDST;
  memory::freeGlobal(ThreadDSTPtr, "remove DST");
  ThreadDSTPtr = OldDST;
}

extern "C" {

// init
void __kmpc_dispatch_init_4(IdentTy *loc, int32_t tid, int32_t schedule,
                            int32_t lb, int32_t ub, int32_t st, int32_t chunk) {
  DynamicScheduleTracker *DST = pushDST();
  omptarget_nvptx_LoopSupport<int32_t, int32_t>::dispatch_init(
      loc, tid, (kmp_sched_t)schedule, lb, ub, st, chunk, DST);
}

void __kmpc_dispatch_init_4u(IdentTy *loc, int32_t tid, int32_t schedule,
                             uint32_t lb, uint32_t ub, int32_t st,
                             int32_t chunk) {
  DynamicScheduleTracker *DST = pushDST();
  omptarget_nvptx_LoopSupport<uint32_t, int32_t>::dispatch_init(
      loc, tid, (kmp_sched_t)schedule, lb, ub, st, chunk, DST);
}

void __kmpc_dispatch_init_8(IdentTy *loc, int32_t tid, int32_t schedule,
                            int64_t lb, int64_t ub, int64_t st, int64_t chunk) {
  DynamicScheduleTracker *DST = pushDST();
  omptarget_nvptx_LoopSupport<int64_t, int64_t>::dispatch_init(
      loc, tid, (kmp_sched_t)schedule, lb, ub, st, chunk, DST);
}

void __kmpc_dispatch_init_8u(IdentTy *loc, int32_t tid, int32_t schedule,
                             uint64_t lb, uint64_t ub, int64_t st,
                             int64_t chunk) {
  DynamicScheduleTracker *DST = pushDST();
  omptarget_nvptx_LoopSupport<uint64_t, int64_t>::dispatch_init(
      loc, tid, (kmp_sched_t)schedule, lb, ub, st, chunk, DST);
}

// next
int __kmpc_dispatch_next_4(IdentTy *loc, int32_t tid, int32_t *p_last,
                           int32_t *p_lb, int32_t *p_ub, int32_t *p_st) {
  DynamicScheduleTracker *DST = peekDST();
  return omptarget_nvptx_LoopSupport<int32_t, int32_t>::dispatch_next(
      loc, tid, p_last, p_lb, p_ub, p_st, DST);
}

int __kmpc_dispatch_next_4u(IdentTy *loc, int32_t tid, int32_t *p_last,
                            uint32_t *p_lb, uint32_t *p_ub, int32_t *p_st) {
  DynamicScheduleTracker *DST = peekDST();
  return omptarget_nvptx_LoopSupport<uint32_t, int32_t>::dispatch_next(
      loc, tid, p_last, p_lb, p_ub, p_st, DST);
}

int __kmpc_dispatch_next_8(IdentTy *loc, int32_t tid, int32_t *p_last,
                           int64_t *p_lb, int64_t *p_ub, int64_t *p_st) {
  DynamicScheduleTracker *DST = peekDST();
  return omptarget_nvptx_LoopSupport<int64_t, int64_t>::dispatch_next(
      loc, tid, p_last, p_lb, p_ub, p_st, DST);
}

int __kmpc_dispatch_next_8u(IdentTy *loc, int32_t tid, int32_t *p_last,
                            uint64_t *p_lb, uint64_t *p_ub, int64_t *p_st) {
  DynamicScheduleTracker *DST = peekDST();
  return omptarget_nvptx_LoopSupport<uint64_t, int64_t>::dispatch_next(
      loc, tid, p_last, p_lb, p_ub, p_st, DST);
}

// fini
void __kmpc_dispatch_fini_4(IdentTy *loc, int32_t tid) {
  omptarget_nvptx_LoopSupport<int32_t, int32_t>::dispatch_fini();
  popDST();
}

void __kmpc_dispatch_fini_4u(IdentTy *loc, int32_t tid) {
  omptarget_nvptx_LoopSupport<uint32_t, int32_t>::dispatch_fini();
  popDST();
}

void __kmpc_dispatch_fini_8(IdentTy *loc, int32_t tid) {
  omptarget_nvptx_LoopSupport<int64_t, int64_t>::dispatch_fini();
  popDST();
}

void __kmpc_dispatch_fini_8u(IdentTy *loc, int32_t tid) {
  omptarget_nvptx_LoopSupport<uint64_t, int64_t>::dispatch_fini();
  popDST();
}

////////////////////////////////////////////////////////////////////////////////
// KMP interface implementation (static loops)
////////////////////////////////////////////////////////////////////////////////

void __kmpc_for_static_init_4(IdentTy *loc, int32_t global_tid,
                              int32_t schedtype, int32_t *plastiter,
                              int32_t *plower, int32_t *pupper,
                              int32_t *pstride, int32_t incr, int32_t chunk) {
  omptarget_nvptx_LoopSupport<int32_t, int32_t>::for_static_init(
      global_tid, schedtype, plastiter, plower, pupper, pstride, chunk,
      mapping::isSPMDMode());
}

void __kmpc_for_static_init_4u(IdentTy *loc, int32_t global_tid,
                               int32_t schedtype, int32_t *plastiter,
                               uint32_t *plower, uint32_t *pupper,
                               int32_t *pstride, int32_t incr, int32_t chunk) {
  omptarget_nvptx_LoopSupport<uint32_t, int32_t>::for_static_init(
      global_tid, schedtype, plastiter, plower, pupper, pstride, chunk,
      mapping::isSPMDMode());
}

void __kmpc_for_static_init_8(IdentTy *loc, int32_t global_tid,
                              int32_t schedtype, int32_t *plastiter,
                              int64_t *plower, int64_t *pupper,
                              int64_t *pstride, int64_t incr, int64_t chunk) {
  omptarget_nvptx_LoopSupport<int64_t, int64_t>::for_static_init(
      global_tid, schedtype, plastiter, plower, pupper, pstride, chunk,
      mapping::isSPMDMode());
}

void __kmpc_for_static_init_8u(IdentTy *loc, int32_t global_tid,
                               int32_t schedtype, int32_t *plastiter,
                               uint64_t *plower, uint64_t *pupper,
                               int64_t *pstride, int64_t incr, int64_t chunk) {
  omptarget_nvptx_LoopSupport<uint64_t, int64_t>::for_static_init(
      global_tid, schedtype, plastiter, plower, pupper, pstride, chunk,
      mapping::isSPMDMode());
}

void __kmpc_for_static_fini(IdentTy *loc, int32_t global_tid) {}
}

#pragma omp end declare target
