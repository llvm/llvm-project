//===------------ loop.cu - NVPTX OpenMP loop constructs --------- CUDA -*-===//
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

#include "omptarget-nvptx.h"
#include "target_impl.h"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// template class that encapsulate all the helper functions
//
// T is loop iteration type (32 | 64)  (unsigned | signed)
// ST is the signed version of T
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <typename T, typename ST> class omptarget_nvptx_LoopSupport {
public:
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
  INLINE static void ForStaticChunk(int &last, T &lb, T &ub, ST &stride,
                                    ST chunk, T entityId, T numberOfEntities) {
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
  INLINE static void ForStaticNoChunk(int &last, T &lb, T &ub, ST &stride,
                                      ST &chunk, T entityId,
                                      T numberOfEntities) {
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

  INLINE static void for_static_init(int32_t gtid, int32_t schedtype,
                                     int32_t *plastiter, T *plower, T *pupper,
                                     ST *pstride, ST chunk,
                                     bool IsSPMDExecutionMode) {
    // When IsRuntimeUninitialized is true, we assume that the caller is
    // in an L0 parallel region and that all worker threads participate.

    // Assume we are in teams region or that we use a single block
    // per target region
    ST numberOfActiveOMPThreads = GetNumberOfOmpThreads(IsSPMDExecutionMode);

    // All warps that are in excess of the maximum requested, do
    // not execute the loop
    PRINT(LD_LOOP,
          "OMP Thread %d: schedule type %d, chunk size = %lld, mytid "
          "%d, num tids %d\n",
          (int)gtid, (int)schedtype, (long long)chunk, (int)gtid,
          (int)numberOfActiveOMPThreads);
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
        ForStaticChunk(lastiter, lb, ub, stride, chunk, GetOmpTeamId(),
                       GetNumberOfOmpTeams());
        break;
      } // note: if chunk <=0, use nochunk
    }
    case kmp_sched_distr_static_nochunk: {
      ForStaticNoChunk(lastiter, lb, ub, stride, chunk, GetOmpTeamId(),
                       GetNumberOfOmpTeams());
      break;
    }
    case kmp_sched_distr_static_chunk_sched_static_chunkone: {
      ForStaticChunk(lastiter, lb, ub, stride, chunk,
                     numberOfActiveOMPThreads * GetOmpTeamId() + gtid,
                     GetNumberOfOmpTeams() * numberOfActiveOMPThreads);
      break;
    }
    default: {
      ASSERT(LT_FUSSY, 0, "unknown schedtype %d", (int)schedtype);
      PRINT(LD_LOOP, "unknown schedtype %d, revert back to static chunk\n",
            (int)schedtype);
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
    PRINT(LD_LOOP,
          "Got sched: Active %d, total %d: lb %lld, ub %lld, stride %lld, last "
          "%d\n",
          (int)numberOfActiveOMPThreads, (int)GetNumberOfWorkersInTeam(),
          (long long)(*plower), (long long)(*pupper), (long long)(*pstride),
          (int)lastiter);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Support for dispatch Init

  INLINE static int OrderedSchedule(kmp_sched_t schedule) {
    return schedule >= kmp_sched_ordered_first &&
           schedule <= kmp_sched_ordered_last;
  }

  INLINE static void dispatch_init(kmp_Ident *loc, int32_t threadId,
                                   kmp_sched_t schedule, T lb, T ub, ST st,
                                   ST chunk) {
    if (checkRuntimeUninitialized(loc)) {
      // In SPMD mode no need to check parallelism level - dynamic scheduling
      // may appear only in L2 parallel regions with lightweight runtime.
      ASSERT0(LT_FUSSY, checkSPMDMode(loc), "Expected non-SPMD mode.");
      return;
    }
    int tid = GetLogicalThreadIdInBlock(checkSPMDMode(loc));
    omptarget_nvptx_TaskDescr *currTaskDescr = getMyTopTaskDescriptor(tid);
    T tnum = GetNumberOfOmpThreads(checkSPMDMode(loc));
    T tripCount = ub - lb + 1; // +1 because ub is inclusive
    ASSERT0(LT_FUSSY, threadId < tnum,
            "current thread is not needed here; error");

    /* Currently just ignore the monotonic and non-monotonic modifiers
     * (the compiler isn't producing them * yet anyway).
     * When it is we'll want to look at them somewhere here and use that
     * information to add to our schedule choice. We shouldn't need to pass
     * them on, they merely affect which schedule we can legally choose for
     * various dynamic cases. (In paritcular, whether or not a stealing scheme
     * is legal).
     */
    schedule = SCHEDULE_WITHOUT_MODIFIERS(schedule);

    // Process schedule.
    if (tnum == 1 || tripCount <= 1 || OrderedSchedule(schedule)) {
      if (OrderedSchedule(schedule))
        __kmpc_barrier(loc, threadId);
      PRINT(LD_LOOP,
            "go sequential as tnum=%ld, trip count %lld, ordered sched=%d\n",
            (long)tnum, (long long)tripCount, (int)schedule);
      schedule = kmp_sched_static_chunk;
      chunk = tripCount; // one thread gets the whole loop
    } else if (schedule == kmp_sched_runtime) {
      // process runtime
      omp_sched_t rtSched = currTaskDescr->GetRuntimeSched();
      chunk = currTaskDescr->RuntimeChunkSize();
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
      PRINT(LD_LOOP, "Runtime sched is %d with chunk %lld\n", (int)schedule,
            (long long)chunk);
    } else if (schedule == kmp_sched_auto) {
      schedule = kmp_sched_static_chunk;
      chunk = 1;
      PRINT(LD_LOOP, "Auto sched is %d with chunk %lld\n", (int)schedule,
            (long long)chunk);
    } else {
      PRINT(LD_LOOP, "Dyn sched is %d with chunk %lld\n", (int)schedule,
            (long long)chunk);
      ASSERT(LT_FUSSY,
             schedule == kmp_sched_dynamic || schedule == kmp_sched_guided,
             "unknown schedule %d & chunk %lld\n", (int)schedule,
             (long long)chunk);
    }

    // init schedules
    if (schedule == kmp_sched_static_chunk) {
      ASSERT0(LT_FUSSY, chunk > 0, "bad chunk value");
      // save sched state
      omptarget_nvptx_threadPrivateContext->ScheduleType(tid) = schedule;
      // save ub
      omptarget_nvptx_threadPrivateContext->LoopUpperBound(tid) = ub;
      // compute static chunk
      ST stride;
      int lastiter = 0;
      ForStaticChunk(lastiter, lb, ub, stride, chunk, threadId, tnum);
      // save computed params
      omptarget_nvptx_threadPrivateContext->Chunk(tid) = chunk;
      omptarget_nvptx_threadPrivateContext->NextLowerBound(tid) = lb;
      omptarget_nvptx_threadPrivateContext->Stride(tid) = stride;
      PRINT(LD_LOOP,
            "dispatch init (static chunk) : num threads = %d, ub =  %" PRId64
            ", next lower bound = %llu, stride = %llu\n",
            (int)tnum,
            omptarget_nvptx_threadPrivateContext->LoopUpperBound(tid),
            (unsigned long long)
                omptarget_nvptx_threadPrivateContext->NextLowerBound(tid),
            (unsigned long long)omptarget_nvptx_threadPrivateContext->Stride(
                tid));
    } else if (schedule == kmp_sched_static_balanced_chunk) {
      ASSERT0(LT_FUSSY, chunk > 0, "bad chunk value");
      // save sched state
      omptarget_nvptx_threadPrivateContext->ScheduleType(tid) = schedule;
      // save ub
      omptarget_nvptx_threadPrivateContext->LoopUpperBound(tid) = ub;
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
      omptarget_nvptx_threadPrivateContext->Chunk(tid) = chunk;
      omptarget_nvptx_threadPrivateContext->NextLowerBound(tid) = lb;
      omptarget_nvptx_threadPrivateContext->Stride(tid) = stride;
      PRINT(LD_LOOP,
            "dispatch init (static chunk) : num threads = %d, ub =  %" PRId64
            ", next lower bound = %llu, stride = %llu\n",
            (int)tnum,
            omptarget_nvptx_threadPrivateContext->LoopUpperBound(tid),
            (unsigned long long)
                omptarget_nvptx_threadPrivateContext->NextLowerBound(tid),
            (unsigned long long)omptarget_nvptx_threadPrivateContext->Stride(
                tid));
    } else if (schedule == kmp_sched_static_nochunk) {
      ASSERT0(LT_FUSSY, chunk == 0, "bad chunk value");
      // save sched state
      omptarget_nvptx_threadPrivateContext->ScheduleType(tid) = schedule;
      // save ub
      omptarget_nvptx_threadPrivateContext->LoopUpperBound(tid) = ub;
      // compute static chunk
      ST stride;
      int lastiter = 0;
      ForStaticNoChunk(lastiter, lb, ub, stride, chunk, threadId, tnum);
      // save computed params
      omptarget_nvptx_threadPrivateContext->Chunk(tid) = chunk;
      omptarget_nvptx_threadPrivateContext->NextLowerBound(tid) = lb;
      omptarget_nvptx_threadPrivateContext->Stride(tid) = stride;
      PRINT(LD_LOOP,
            "dispatch init (static nochunk) : num threads = %d, ub = %" PRId64
            ", next lower bound = %llu, stride = %llu\n",
            (int)tnum,
            omptarget_nvptx_threadPrivateContext->LoopUpperBound(tid),
            (unsigned long long)
                omptarget_nvptx_threadPrivateContext->NextLowerBound(tid),
            (unsigned long long)omptarget_nvptx_threadPrivateContext->Stride(
                tid));
    } else if (schedule == kmp_sched_dynamic || schedule == kmp_sched_guided) {
      // save data
      omptarget_nvptx_threadPrivateContext->ScheduleType(tid) = schedule;
      if (chunk < 1)
        chunk = 1;
      omptarget_nvptx_threadPrivateContext->Chunk(tid) = chunk;
      omptarget_nvptx_threadPrivateContext->LoopUpperBound(tid) = ub;
      omptarget_nvptx_threadPrivateContext->NextLowerBound(tid) = lb;
      __kmpc_barrier(loc, threadId);
      if (tid == 0) {
        omptarget_nvptx_threadPrivateContext->Cnt() = 0;
        __threadfence_block();
      }
      __kmpc_barrier(loc, threadId);
      PRINT(LD_LOOP,
            "dispatch init (dyn) : num threads = %d, lb = %llu, ub = %" PRId64
            ", chunk %" PRIu64 "\n",
            (int)tnum,
            (unsigned long long)
                omptarget_nvptx_threadPrivateContext->NextLowerBound(tid),
            omptarget_nvptx_threadPrivateContext->LoopUpperBound(tid),
            omptarget_nvptx_threadPrivateContext->Chunk(tid));
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Support for dispatch next

  INLINE static uint64_t Shuffle(__kmpc_impl_lanemask_t active, int64_t val,
                                 int leader) {
    uint32_t lo, hi;
    __kmpc_impl_unpack(val, lo, hi);
    hi = __kmpc_impl_shfl_sync(active, hi, leader);
    lo = __kmpc_impl_shfl_sync(active, lo, leader);
    return __kmpc_impl_pack(lo, hi);
  }

  INLINE static uint64_t NextIter() {
    __kmpc_impl_lanemask_t active = __kmpc_impl_activemask();
    uint32_t leader = __kmpc_impl_ffs(active) - 1;
    uint32_t change = __kmpc_impl_popc(active);
    __kmpc_impl_lanemask_t lane_mask_lt = __kmpc_impl_lanemask_lt();
    unsigned int rank = __kmpc_impl_popc(active & lane_mask_lt);
    uint64_t warp_res;
    if (rank == 0) {
      warp_res = atomicAdd(
          (unsigned long long *)&omptarget_nvptx_threadPrivateContext->Cnt(),
          change);
    }
    warp_res = Shuffle(active, warp_res, leader);
    return warp_res + rank;
  }

  INLINE static int DynamicNextChunk(T &lb, T &ub, T chunkSize,
                                     T loopLowerBound, T loopUpperBound) {
    T N = NextIter();
    lb = loopLowerBound + N * chunkSize;
    ub = lb + chunkSize - 1;  // Clang uses i <= ub

    // 3 result cases:
    //  a. lb and ub < loopUpperBound --> NOT_FINISHED
    //  b. lb < loopUpperBound and ub >= loopUpperBound: last chunk -->
    //  NOT_FINISHED
    //  c. lb and ub >= loopUpperBound: empty chunk --> FINISHED
    // a.
    if (lb <= loopUpperBound && ub < loopUpperBound) {
      PRINT(LD_LOOPD, "lb %lld, ub %lld, loop ub %lld; not finished\n",
            (long long)lb, (long long)ub, (long long)loopUpperBound);
      return NOT_FINISHED;
    }
    // b.
    if (lb <= loopUpperBound) {
      PRINT(LD_LOOPD, "lb %lld, ub %lld, loop ub %lld; clip to loop ub\n",
            (long long)lb, (long long)ub, (long long)loopUpperBound);
      ub = loopUpperBound;
      return LAST_CHUNK;
    }
    // c. if we are here, we are in case 'c'
    lb = loopUpperBound + 2;
    ub = loopUpperBound + 1;
    PRINT(LD_LOOPD, "lb %lld, ub %lld, loop ub %lld; finished\n", (long long)lb,
          (long long)ub, (long long)loopUpperBound);
    return FINISHED;
  }

  INLINE static int dispatch_next(kmp_Ident *loc, int32_t gtid, int32_t *plast,
                                  T *plower, T *pupper, ST *pstride) {
    if (checkRuntimeUninitialized(loc)) {
      // In SPMD mode no need to check parallelism level - dynamic scheduling
      // may appear only in L2 parallel regions with lightweight runtime.
      ASSERT0(LT_FUSSY, checkSPMDMode(loc), "Expected non-SPMD mode.");
      if (*plast)
        return DISPATCH_FINISHED;
      *plast = 1;
      return DISPATCH_NOTFINISHED;
    }
    // ID of a thread in its own warp

    // automatically selects thread or warp ID based on selected implementation
    int tid = GetLogicalThreadIdInBlock(checkSPMDMode(loc));
    ASSERT0(LT_FUSSY, gtid < GetNumberOfOmpThreads(checkSPMDMode(loc)),
            "current thread is not needed here; error");
    // retrieve schedule
    kmp_sched_t schedule =
        omptarget_nvptx_threadPrivateContext->ScheduleType(tid);

    // xxx reduce to one
    if (schedule == kmp_sched_static_chunk ||
        schedule == kmp_sched_static_nochunk) {
      T myLb = omptarget_nvptx_threadPrivateContext->NextLowerBound(tid);
      T ub = omptarget_nvptx_threadPrivateContext->LoopUpperBound(tid);
      // finished?
      if (myLb > ub) {
        PRINT(LD_LOOP, "static loop finished with myLb %lld, ub %lld\n",
              (long long)myLb, (long long)ub);
        return DISPATCH_FINISHED;
      }
      // not finished, save current bounds
      ST chunk = omptarget_nvptx_threadPrivateContext->Chunk(tid);
      *plower = myLb;
      T myUb = myLb + chunk - 1; // Clang uses i <= ub
      if (myUb > ub)
        myUb = ub;
      *pupper = myUb;
      *plast = (int32_t)(myUb == ub);

      // increment next lower bound by the stride
      ST stride = omptarget_nvptx_threadPrivateContext->Stride(tid);
      omptarget_nvptx_threadPrivateContext->NextLowerBound(tid) = myLb + stride;
      PRINT(LD_LOOP, "static loop continues with myLb %lld, myUb %lld\n",
            (long long)*plower, (long long)*pupper);
      return DISPATCH_NOTFINISHED;
    }
    ASSERT0(LT_FUSSY,
            schedule == kmp_sched_dynamic || schedule == kmp_sched_guided,
            "bad sched");
    T myLb, myUb;
    int finished = DynamicNextChunk(
        myLb, myUb, omptarget_nvptx_threadPrivateContext->Chunk(tid),
        omptarget_nvptx_threadPrivateContext->NextLowerBound(tid),
        omptarget_nvptx_threadPrivateContext->LoopUpperBound(tid));

    if (finished == FINISHED)
      return DISPATCH_FINISHED;

    // not finished (either not finished or last chunk)
    *plast = (int32_t)(finished == LAST_CHUNK);
    *plower = myLb;
    *pupper = myUb;
    *pstride = 1;

    PRINT(LD_LOOP,
          "Got sched: active %d, total %d: lb %lld, ub %lld, stride = %lld, "
          "last %d\n",
          (int)GetNumberOfOmpThreads(isSPMDMode()),
          (int)GetNumberOfWorkersInTeam(), (long long)*plower,
          (long long)*pupper, (long long)*pstride, (int)*plast);
    return DISPATCH_NOTFINISHED;
  }

  INLINE static void dispatch_fini() {
    // nothing
  }

  ////////////////////////////////////////////////////////////////////////////////
  // end of template class that encapsulate all the helper functions
  ////////////////////////////////////////////////////////////////////////////////
};

////////////////////////////////////////////////////////////////////////////////
// KMP interface implementation (dyn loops)
////////////////////////////////////////////////////////////////////////////////

// init
EXTERN void __kmpc_dispatch_init_4(kmp_Ident *loc, int32_t tid,
                                   int32_t schedule, int32_t lb, int32_t ub,
                                   int32_t st, int32_t chunk) {
  PRINT0(LD_IO, "call kmpc_dispatch_init_4\n");
  omptarget_nvptx_LoopSupport<int32_t, int32_t>::dispatch_init(
      loc, tid, (kmp_sched_t)schedule, lb, ub, st, chunk);
}

EXTERN void __kmpc_dispatch_init_4u(kmp_Ident *loc, int32_t tid,
                                    int32_t schedule, uint32_t lb, uint32_t ub,
                                    int32_t st, int32_t chunk) {
  PRINT0(LD_IO, "call kmpc_dispatch_init_4u\n");
  omptarget_nvptx_LoopSupport<uint32_t, int32_t>::dispatch_init(
      loc, tid, (kmp_sched_t)schedule, lb, ub, st, chunk);
}

EXTERN void __kmpc_dispatch_init_8(kmp_Ident *loc, int32_t tid,
                                   int32_t schedule, int64_t lb, int64_t ub,
                                   int64_t st, int64_t chunk) {
  PRINT0(LD_IO, "call kmpc_dispatch_init_8\n");
  omptarget_nvptx_LoopSupport<int64_t, int64_t>::dispatch_init(
      loc, tid, (kmp_sched_t)schedule, lb, ub, st, chunk);
}

EXTERN void __kmpc_dispatch_init_8u(kmp_Ident *loc, int32_t tid,
                                    int32_t schedule, uint64_t lb, uint64_t ub,
                                    int64_t st, int64_t chunk) {
  PRINT0(LD_IO, "call kmpc_dispatch_init_8u\n");
  omptarget_nvptx_LoopSupport<uint64_t, int64_t>::dispatch_init(
      loc, tid, (kmp_sched_t)schedule, lb, ub, st, chunk);
}

// next
EXTERN int __kmpc_dispatch_next_4(kmp_Ident *loc, int32_t tid, int32_t *p_last,
                                  int32_t *p_lb, int32_t *p_ub, int32_t *p_st) {
  PRINT0(LD_IO, "call kmpc_dispatch_next_4\n");
  return omptarget_nvptx_LoopSupport<int32_t, int32_t>::dispatch_next(
      loc, tid, p_last, p_lb, p_ub, p_st);
}

EXTERN int __kmpc_dispatch_next_4u(kmp_Ident *loc, int32_t tid,
                                   int32_t *p_last, uint32_t *p_lb,
                                   uint32_t *p_ub, int32_t *p_st) {
  PRINT0(LD_IO, "call kmpc_dispatch_next_4u\n");
  return omptarget_nvptx_LoopSupport<uint32_t, int32_t>::dispatch_next(
      loc, tid, p_last, p_lb, p_ub, p_st);
}

EXTERN int __kmpc_dispatch_next_8(kmp_Ident *loc, int32_t tid, int32_t *p_last,
                                  int64_t *p_lb, int64_t *p_ub, int64_t *p_st) {
  PRINT0(LD_IO, "call kmpc_dispatch_next_8\n");
  return omptarget_nvptx_LoopSupport<int64_t, int64_t>::dispatch_next(
      loc, tid, p_last, p_lb, p_ub, p_st);
}

EXTERN int __kmpc_dispatch_next_8u(kmp_Ident *loc, int32_t tid,
                                   int32_t *p_last, uint64_t *p_lb,
                                   uint64_t *p_ub, int64_t *p_st) {
  PRINT0(LD_IO, "call kmpc_dispatch_next_8u\n");
  return omptarget_nvptx_LoopSupport<uint64_t, int64_t>::dispatch_next(
      loc, tid, p_last, p_lb, p_ub, p_st);
}

// fini
EXTERN void __kmpc_dispatch_fini_4(kmp_Ident *loc, int32_t tid) {
  PRINT0(LD_IO, "call kmpc_dispatch_fini_4\n");
  omptarget_nvptx_LoopSupport<int32_t, int32_t>::dispatch_fini();
}

EXTERN void __kmpc_dispatch_fini_4u(kmp_Ident *loc, int32_t tid) {
  PRINT0(LD_IO, "call kmpc_dispatch_fini_4u\n");
  omptarget_nvptx_LoopSupport<uint32_t, int32_t>::dispatch_fini();
}

EXTERN void __kmpc_dispatch_fini_8(kmp_Ident *loc, int32_t tid) {
  PRINT0(LD_IO, "call kmpc_dispatch_fini_8\n");
  omptarget_nvptx_LoopSupport<int64_t, int64_t>::dispatch_fini();
}

EXTERN void __kmpc_dispatch_fini_8u(kmp_Ident *loc, int32_t tid) {
  PRINT0(LD_IO, "call kmpc_dispatch_fini_8u\n");
  omptarget_nvptx_LoopSupport<uint64_t, int64_t>::dispatch_fini();
}

////////////////////////////////////////////////////////////////////////////////
// KMP interface implementation (static loops)
////////////////////////////////////////////////////////////////////////////////

EXTERN void __kmpc_for_static_init_4(kmp_Ident *loc, int32_t global_tid,
                                     int32_t schedtype, int32_t *plastiter,
                                     int32_t *plower, int32_t *pupper,
                                     int32_t *pstride, int32_t incr,
                                     int32_t chunk) {
  PRINT0(LD_IO, "call kmpc_for_static_init_4\n");
  omptarget_nvptx_LoopSupport<int32_t, int32_t>::for_static_init(
      global_tid, schedtype, plastiter, plower, pupper, pstride, chunk,
      checkSPMDMode(loc));
}

EXTERN void __kmpc_for_static_init_4u(kmp_Ident *loc, int32_t global_tid,
                                      int32_t schedtype, int32_t *plastiter,
                                      uint32_t *plower, uint32_t *pupper,
                                      int32_t *pstride, int32_t incr,
                                      int32_t chunk) {
  PRINT0(LD_IO, "call kmpc_for_static_init_4u\n");
  omptarget_nvptx_LoopSupport<uint32_t, int32_t>::for_static_init(
      global_tid, schedtype, plastiter, plower, pupper, pstride, chunk,
      checkSPMDMode(loc));
}

EXTERN void __kmpc_for_static_init_8(kmp_Ident *loc, int32_t global_tid,
                                     int32_t schedtype, int32_t *plastiter,
                                     int64_t *plower, int64_t *pupper,
                                     int64_t *pstride, int64_t incr,
                                     int64_t chunk) {
  PRINT0(LD_IO, "call kmpc_for_static_init_8\n");
  omptarget_nvptx_LoopSupport<int64_t, int64_t>::for_static_init(
      global_tid, schedtype, plastiter, plower, pupper, pstride, chunk,
      checkSPMDMode(loc));
}

EXTERN void __kmpc_for_static_init_8u(kmp_Ident *loc, int32_t global_tid,
                                      int32_t schedtype, int32_t *plastiter,
                                      uint64_t *plower, uint64_t *pupper,
                                      int64_t *pstride, int64_t incr,
                                      int64_t chunk) {
  PRINT0(LD_IO, "call kmpc_for_static_init_8u\n");
  omptarget_nvptx_LoopSupport<uint64_t, int64_t>::for_static_init(
      global_tid, schedtype, plastiter, plower, pupper, pstride, chunk,
      checkSPMDMode(loc));
}

EXTERN
void __kmpc_for_static_init_4_simple_spmd(kmp_Ident *loc, int32_t global_tid,
                                          int32_t schedtype, int32_t *plastiter,
                                          int32_t *plower, int32_t *pupper,
                                          int32_t *pstride, int32_t incr,
                                          int32_t chunk) {
  PRINT0(LD_IO, "call kmpc_for_static_init_4_simple_spmd\n");
  omptarget_nvptx_LoopSupport<int32_t, int32_t>::for_static_init(
      global_tid, schedtype, plastiter, plower, pupper, pstride, chunk,
      /*IsSPMDExecutionMode=*/true);
}

EXTERN
void __kmpc_for_static_init_4u_simple_spmd(kmp_Ident *loc, int32_t global_tid,
                                           int32_t schedtype,
                                           int32_t *plastiter, uint32_t *plower,
                                           uint32_t *pupper, int32_t *pstride,
                                           int32_t incr, int32_t chunk) {
  PRINT0(LD_IO, "call kmpc_for_static_init_4u_simple_spmd\n");
  omptarget_nvptx_LoopSupport<uint32_t, int32_t>::for_static_init(
      global_tid, schedtype, plastiter, plower, pupper, pstride, chunk,
      /*IsSPMDExecutionMode=*/true);
}

EXTERN
void __kmpc_for_static_init_8_simple_spmd(kmp_Ident *loc, int32_t global_tid,
                                          int32_t schedtype, int32_t *plastiter,
                                          int64_t *plower, int64_t *pupper,
                                          int64_t *pstride, int64_t incr,
                                          int64_t chunk) {
  PRINT0(LD_IO, "call kmpc_for_static_init_8_simple_spmd\n");
  omptarget_nvptx_LoopSupport<int64_t, int64_t>::for_static_init(
      global_tid, schedtype, plastiter, plower, pupper, pstride, chunk,
      /*IsSPMDExecutionMode=*/true);
}

EXTERN
void __kmpc_for_static_init_8u_simple_spmd(kmp_Ident *loc, int32_t global_tid,
                                           int32_t schedtype,
                                           int32_t *plastiter, uint64_t *plower,
                                           uint64_t *pupper, int64_t *pstride,
                                           int64_t incr, int64_t chunk) {
  PRINT0(LD_IO, "call kmpc_for_static_init_8u_simple_spmd\n");
  omptarget_nvptx_LoopSupport<uint64_t, int64_t>::for_static_init(
      global_tid, schedtype, plastiter, plower, pupper, pstride, chunk,
      /*IsSPMDExecutionMode=*/true);
}

EXTERN
void __kmpc_for_static_init_4_simple_generic(
    kmp_Ident *loc, int32_t global_tid, int32_t schedtype, int32_t *plastiter,
    int32_t *plower, int32_t *pupper, int32_t *pstride, int32_t incr,
    int32_t chunk) {
  PRINT0(LD_IO, "call kmpc_for_static_init_4_simple_generic\n");
  omptarget_nvptx_LoopSupport<int32_t, int32_t>::for_static_init(
      global_tid, schedtype, plastiter, plower, pupper, pstride, chunk,
      /*IsSPMDExecutionMode=*/false);
}

EXTERN
void __kmpc_for_static_init_4u_simple_generic(
    kmp_Ident *loc, int32_t global_tid, int32_t schedtype, int32_t *plastiter,
    uint32_t *plower, uint32_t *pupper, int32_t *pstride, int32_t incr,
    int32_t chunk) {
  PRINT0(LD_IO, "call kmpc_for_static_init_4u_simple_generic\n");
  omptarget_nvptx_LoopSupport<uint32_t, int32_t>::for_static_init(
      global_tid, schedtype, plastiter, plower, pupper, pstride, chunk,
      /*IsSPMDExecutionMode=*/false);
}

EXTERN
void __kmpc_for_static_init_8_simple_generic(
    kmp_Ident *loc, int32_t global_tid, int32_t schedtype, int32_t *plastiter,
    int64_t *plower, int64_t *pupper, int64_t *pstride, int64_t incr,
    int64_t chunk) {
  PRINT0(LD_IO, "call kmpc_for_static_init_8_simple_generic\n");
  omptarget_nvptx_LoopSupport<int64_t, int64_t>::for_static_init(
      global_tid, schedtype, plastiter, plower, pupper, pstride, chunk,
      /*IsSPMDExecutionMode=*/false);
}

EXTERN
void __kmpc_for_static_init_8u_simple_generic(
    kmp_Ident *loc, int32_t global_tid, int32_t schedtype, int32_t *plastiter,
    uint64_t *plower, uint64_t *pupper, int64_t *pstride, int64_t incr,
    int64_t chunk) {
  PRINT0(LD_IO, "call kmpc_for_static_init_8u_simple_generic\n");
  omptarget_nvptx_LoopSupport<uint64_t, int64_t>::for_static_init(
      global_tid, schedtype, plastiter, plower, pupper, pstride, chunk,
      /*IsSPMDExecutionMode=*/false);
}

EXTERN void __kmpc_for_static_fini(kmp_Ident *loc, int32_t global_tid) {
  PRINT0(LD_IO, "call kmpc_for_static_fini\n");
}

namespace {
INLINE void syncWorkersInGenericMode(uint32_t NumThreads) {
  int NumWarps = ((NumThreads + WARPSIZE - 1) / WARPSIZE);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  // On Volta and newer architectures we require that all lanes in
  // a warp (at least, all present for the kernel launch) participate in the
  // barrier.  This is enforced when launching the parallel region.  An
  // exception is when there are < WARPSIZE workers.  In this case only 1 worker
  // is started, so we don't need a barrier.
  if (NumThreads > 1) {
#endif
    __kmpc_impl_named_sync(L1_BARRIER, WARPSIZE * NumWarps);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  }
#endif
}
}; // namespace

EXTERN void __kmpc_reduce_conditional_lastprivate(kmp_Ident *loc, int32_t gtid,
                                                  int32_t varNum, void *array) {
  PRINT0(LD_IO, "call to __kmpc_reduce_conditional_lastprivate(...)\n");
  ASSERT0(LT_FUSSY, checkRuntimeInitialized(loc),
          "Expected non-SPMD mode + initialized runtime.");

  omptarget_nvptx_TeamDescr &teamDescr = getMyTeamDescriptor();
  uint32_t NumThreads = GetNumberOfOmpThreads(checkSPMDMode(loc));
  uint64_t *Buffer = teamDescr.getLastprivateIterBuffer();
  for (unsigned i = 0; i < varNum; i++) {
    // Reset buffer.
    if (gtid == 0)
      *Buffer = 0; // Reset to minimum loop iteration value.

    // Barrier.
    syncWorkersInGenericMode(NumThreads);

    // Atomic max of iterations.
    uint64_t *varArray = (uint64_t *)array;
    uint64_t elem = varArray[i];
    (void)atomicMax((unsigned long long int *)Buffer,
                    (unsigned long long int)elem);

    // Barrier.
    syncWorkersInGenericMode(NumThreads);

    // Read max value and update thread private array.
    varArray[i] = *Buffer;

    // Barrier.
    syncWorkersInGenericMode(NumThreads);
  }
}
