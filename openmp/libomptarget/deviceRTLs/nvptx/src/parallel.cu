//===---- parallel.cu - NVPTX OpenMP parallel implementation ----- CUDA -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Parallel implemention in the GPU. Here is the pattern:
//
//    while (not finished) {
//
//    if (master) {
//      sequential code, decide which par loop to do, or if finished
//     __kmpc_kernel_prepare_parallel() // exec by master only
//    }
//    syncthreads // A
//    __kmpc_kernel_parallel() // exec by all
//    if (this thread is included in the parallel) {
//      switch () for all parallel loops
//      __kmpc_kernel_end_parallel() // exec only by threads in parallel
//    }
//
//
//    The reason we don't exec end_parallel for the threads not included
//    in the parallel loop is that for each barrier in the parallel
//    region, these non-included threads will cycle through the
//    syncthread A. Thus they must preserve their current threadId that
//    is larger than thread in team.
//
//    To make a long story short...
//
//===----------------------------------------------------------------------===//

#include "omptarget-nvptx.h"

typedef struct ConvergentSimdJob {
  omptarget_nvptx_TaskDescr taskDescr;
  omptarget_nvptx_TaskDescr *convHeadTaskDescr;
  uint16_t slimForNextSimd;
} ConvergentSimdJob;

////////////////////////////////////////////////////////////////////////////////
// support for convergent simd (team of threads in a warp only)
////////////////////////////////////////////////////////////////////////////////
EXTERN bool __kmpc_kernel_convergent_simd(void *buffer, uint32_t Mask,
                                          bool *IsFinal, int32_t *LaneSource,
                                          int32_t *LaneId, int32_t *NumLanes) {
  PRINT0(LD_IO, "call to __kmpc_kernel_convergent_simd\n");
  uint32_t ConvergentMask = Mask;
  int32_t ConvergentSize = __popc(ConvergentMask);
  uint32_t WorkRemaining = ConvergentMask >> (*LaneSource + 1);
  *LaneSource += __ffs(WorkRemaining);
  *IsFinal = __popc(WorkRemaining) == 1;
  uint32_t lanemask_lt;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lanemask_lt));
  *LaneId = __popc(ConvergentMask & lanemask_lt);

  int threadId = GetLogicalThreadIdInBlock(isSPMDMode());
  int sourceThreadId = (threadId & ~(WARPSIZE - 1)) + *LaneSource;

  ConvergentSimdJob *job = (ConvergentSimdJob *)buffer;
  int32_t SimdLimit =
      omptarget_nvptx_threadPrivateContext->SimdLimitForNextSimd(threadId);
  job->slimForNextSimd = SimdLimit;

  int32_t SimdLimitSource = __SHFL_SYNC(Mask, SimdLimit, *LaneSource);
  // reset simdlimit to avoid propagating to successive #simd
  if (SimdLimitSource > 0 && threadId == sourceThreadId)
    omptarget_nvptx_threadPrivateContext->SimdLimitForNextSimd(threadId) = 0;

  // We cannot have more than the # of convergent threads.
  if (SimdLimitSource > 0)
    *NumLanes = min(ConvergentSize, SimdLimitSource);
  else
    *NumLanes = ConvergentSize;
  ASSERT(LT_FUSSY, *NumLanes > 0, "bad thread request of %d threads",
         (int)*NumLanes);

  // Set to true for lanes participating in the simd region.
  bool isActive = false;
  // Initialize state for active threads.
  if (*LaneId < *NumLanes) {
    omptarget_nvptx_TaskDescr *currTaskDescr =
        omptarget_nvptx_threadPrivateContext->GetTopLevelTaskDescr(threadId);
    omptarget_nvptx_TaskDescr *sourceTaskDescr =
        omptarget_nvptx_threadPrivateContext->GetTopLevelTaskDescr(
            sourceThreadId);
    job->convHeadTaskDescr = currTaskDescr;
    // install top descriptor from the thread for which the lanes are working.
    omptarget_nvptx_threadPrivateContext->SetTopLevelTaskDescr(threadId,
                                                               sourceTaskDescr);
    isActive = true;
  }

  // requires a memory fence between threads of a warp
  return isActive;
}

EXTERN void __kmpc_kernel_end_convergent_simd(void *buffer) {
  PRINT0(LD_IO | LD_PAR, "call to __kmpc_kernel_end_convergent_parallel\n");
  // pop stack
  int threadId = GetLogicalThreadIdInBlock(isSPMDMode());
  ConvergentSimdJob *job = (ConvergentSimdJob *)buffer;
  omptarget_nvptx_threadPrivateContext->SimdLimitForNextSimd(threadId) =
      job->slimForNextSimd;
  omptarget_nvptx_threadPrivateContext->SetTopLevelTaskDescr(
      threadId, job->convHeadTaskDescr);
}

typedef struct ConvergentParallelJob {
  omptarget_nvptx_TaskDescr taskDescr;
  omptarget_nvptx_TaskDescr *convHeadTaskDescr;
  uint16_t tnumForNextPar;
} ConvergentParallelJob;

////////////////////////////////////////////////////////////////////////////////
// support for convergent parallelism (team of threads in a warp only)
////////////////////////////////////////////////////////////////////////////////
EXTERN bool __kmpc_kernel_convergent_parallel(void *buffer, uint32_t Mask,
                                              bool *IsFinal,
                                              int32_t *LaneSource) {
  PRINT0(LD_IO, "call to __kmpc_kernel_convergent_parallel\n");
  uint32_t ConvergentMask = Mask;
  int32_t ConvergentSize = __popc(ConvergentMask);
  uint32_t WorkRemaining = ConvergentMask >> (*LaneSource + 1);
  *LaneSource += __ffs(WorkRemaining);
  *IsFinal = __popc(WorkRemaining) == 1;
  uint32_t lanemask_lt;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lanemask_lt));
  uint32_t OmpId = __popc(ConvergentMask & lanemask_lt);

  int threadId = GetLogicalThreadIdInBlock(isSPMDMode());
  int sourceThreadId = (threadId & ~(WARPSIZE - 1)) + *LaneSource;

  ConvergentParallelJob *job = (ConvergentParallelJob *)buffer;
  int32_t NumThreadsClause =
      omptarget_nvptx_threadPrivateContext->NumThreadsForNextParallel(threadId);
  job->tnumForNextPar = NumThreadsClause;

  int32_t NumThreadsSource = __SHFL_SYNC(Mask, NumThreadsClause, *LaneSource);
  // reset numthreads to avoid propagating to successive #parallel
  if (NumThreadsSource > 0 && threadId == sourceThreadId)
    omptarget_nvptx_threadPrivateContext->NumThreadsForNextParallel(threadId) =
        0;

  // We cannot have more than the # of convergent threads.
  uint16_t NumThreads;
  if (NumThreadsSource > 0)
    NumThreads = min(ConvergentSize, NumThreadsSource);
  else
    NumThreads = ConvergentSize;
  ASSERT(LT_FUSSY, NumThreads > 0, "bad thread request of %d threads",
         (int)NumThreads);

  // Set to true for workers participating in the parallel region.
  bool isActive = false;
  // Initialize state for active threads.
  if (OmpId < NumThreads) {
    // init L2 task descriptor and storage for the L1 parallel task descriptor.
    omptarget_nvptx_TaskDescr *newTaskDescr = &job->taskDescr;
    ASSERT0(LT_FUSSY, newTaskDescr, "expected a task descr");
    omptarget_nvptx_TaskDescr *currTaskDescr =
        omptarget_nvptx_threadPrivateContext->GetTopLevelTaskDescr(threadId);
    omptarget_nvptx_TaskDescr *sourceTaskDescr =
        omptarget_nvptx_threadPrivateContext->GetTopLevelTaskDescr(
            sourceThreadId);
    job->convHeadTaskDescr = currTaskDescr;
    newTaskDescr->CopyConvergentParent(sourceTaskDescr, OmpId, NumThreads);
    // install new top descriptor
    omptarget_nvptx_threadPrivateContext->SetTopLevelTaskDescr(threadId,
                                                               newTaskDescr);
    isActive = true;
  }

  // requires a memory fence between threads of a warp
  return isActive;
}

EXTERN void __kmpc_kernel_end_convergent_parallel(void *buffer) {
  PRINT0(LD_IO | LD_PAR, "call to __kmpc_kernel_end_convergent_parallel\n");
  // pop stack
  int threadId = GetLogicalThreadIdInBlock(isSPMDMode());
  ConvergentParallelJob *job = (ConvergentParallelJob *)buffer;
  omptarget_nvptx_threadPrivateContext->SetTopLevelTaskDescr(
      threadId, job->convHeadTaskDescr);
  omptarget_nvptx_threadPrivateContext->NumThreadsForNextParallel(threadId) =
      job->tnumForNextPar;
}

////////////////////////////////////////////////////////////////////////////////
// support for parallel that goes parallel (1 static level only)
////////////////////////////////////////////////////////////////////////////////

INLINE static uint16_t determineNumberOfThreads(uint16_t NumThreadsClause,
                                                uint16_t NThreadsICV,
                                                uint16_t ThreadLimit) {
  uint16_t ThreadsRequested = NThreadsICV;
  if (NumThreadsClause != 0) {
    ThreadsRequested = NumThreadsClause;
  }

  uint16_t ThreadsAvailable = GetNumberOfWorkersInTeam();
  if (ThreadLimit != 0 && ThreadLimit < ThreadsAvailable) {
    ThreadsAvailable = ThreadLimit;
  }

  uint16_t NumThreads = ThreadsAvailable;
  if (ThreadsRequested != 0 && ThreadsRequested < NumThreads) {
    NumThreads = ThreadsRequested;
  }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  // On Volta and newer architectures we require that all lanes in
  // a warp participate in the parallel region.  Round down to a
  // multiple of WARPSIZE since it is legal to do so in OpenMP.
  if (NumThreads < WARPSIZE) {
    NumThreads = 1;
  } else {
    NumThreads = (NumThreads & ~((uint16_t)WARPSIZE - 1));
  }
#endif

  return NumThreads;
}

// This routine is always called by the team master..
EXTERN void __kmpc_kernel_prepare_parallel(void *WorkFn,
                                           int16_t IsOMPRuntimeInitialized) {
  PRINT0(LD_IO, "call to __kmpc_kernel_prepare_parallel\n");
  ASSERT0(LT_FUSSY, IsOMPRuntimeInitialized, "Expected initialized runtime.");

  omptarget_nvptx_workFn = WorkFn;

  // This routine is only called by the team master.  The team master is
  // the first thread of the last warp.  It always has the logical thread
  // id of 0 (since it is a shadow for the first worker thread).
  const int threadId = 0;
  omptarget_nvptx_TaskDescr *currTaskDescr =
      omptarget_nvptx_threadPrivateContext->GetTopLevelTaskDescr(threadId);
  ASSERT0(LT_FUSSY, currTaskDescr, "expected a top task descr");
  ASSERT0(LT_FUSSY, !currTaskDescr->InParallelRegion(),
          "cannot be called in a parallel region.");
  if (currTaskDescr->InParallelRegion()) {
    PRINT0(LD_PAR, "already in parallel: go seq\n");
    return;
  }

  uint16_t &NumThreadsClause =
      omptarget_nvptx_threadPrivateContext->NumThreadsForNextParallel(threadId);

  uint16_t NumThreads =
      determineNumberOfThreads(NumThreadsClause, nThreads, threadLimit);

  if (NumThreadsClause != 0) {
    // Reset request to avoid propagating to successive #parallel
    NumThreadsClause = 0;
  }

  ASSERT(LT_FUSSY, NumThreads > 0, "bad thread request of %d threads",
         (int)NumThreads);
  ASSERT0(LT_FUSSY, GetThreadIdInBlock() == GetMasterThreadID(),
          "only team master can create parallel");

  // Set number of threads on work descriptor.
  omptarget_nvptx_WorkDescr &workDescr = getMyWorkDescriptor();
  workDescr.WorkTaskDescr()->CopyToWorkDescr(currTaskDescr);
  threadsInTeam = NumThreads;
}

// All workers call this function.  Deactivate those not needed.
// Fn - the outlined work function to execute.
// returns True if this thread is active, else False.
//
// Only the worker threads call this routine.
EXTERN bool __kmpc_kernel_parallel(void **WorkFn,
                                   int16_t IsOMPRuntimeInitialized) {
  PRINT0(LD_IO | LD_PAR, "call to __kmpc_kernel_parallel\n");

  ASSERT0(LT_FUSSY, IsOMPRuntimeInitialized, "Expected initialized runtime.");

  // Work function and arguments for L1 parallel region.
  *WorkFn = omptarget_nvptx_workFn;

  // If this is the termination signal from the master, quit early.
  if (!*WorkFn) {
    PRINT0(LD_IO | LD_PAR, "call to __kmpc_kernel_parallel finished\n");
    return false;
  }

  // Only the worker threads call this routine and the master warp
  // never arrives here.  Therefore, use the nvptx thread id.
  int threadId = GetThreadIdInBlock();
  omptarget_nvptx_WorkDescr &workDescr = getMyWorkDescriptor();
  // Set to true for workers participating in the parallel region.
  bool isActive = false;
  // Initialize state for active threads.
  if (threadId < threadsInTeam) {
    // init work descriptor from workdesccr
    omptarget_nvptx_TaskDescr *newTaskDescr =
        omptarget_nvptx_threadPrivateContext->Level1TaskDescr(threadId);
    ASSERT0(LT_FUSSY, newTaskDescr, "expected a task descr");
    newTaskDescr->CopyFromWorkDescr(workDescr.WorkTaskDescr());
    // install new top descriptor
    omptarget_nvptx_threadPrivateContext->SetTopLevelTaskDescr(threadId,
                                                               newTaskDescr);
    // init private from int value
    PRINT(LD_PAR,
          "thread will execute parallel region with id %d in a team of "
          "%d threads\n",
          (int)newTaskDescr->ThreadId(), (int)nThreads);

    isActive = true;
    IncParallelLevel(threadsInTeam != 1);
  }

  return isActive;
}

EXTERN void __kmpc_kernel_end_parallel() {
  // pop stack
  PRINT0(LD_IO | LD_PAR, "call to __kmpc_kernel_end_parallel\n");
  ASSERT0(LT_FUSSY, isRuntimeInitialized(), "Expected initialized runtime.");

  // Only the worker threads call this routine and the master warp
  // never arrives here.  Therefore, use the nvptx thread id.
  int threadId = GetThreadIdInBlock();
  omptarget_nvptx_TaskDescr *currTaskDescr = getMyTopTaskDescriptor(threadId);
  omptarget_nvptx_threadPrivateContext->SetTopLevelTaskDescr(
      threadId, currTaskDescr->GetPrevTaskDescr());

  DecParallelLevel(threadsInTeam != 1);
}

////////////////////////////////////////////////////////////////////////////////
// support for parallel that goes sequential
////////////////////////////////////////////////////////////////////////////////

EXTERN void __kmpc_serialized_parallel(kmp_Ident *loc, uint32_t global_tid) {
  PRINT0(LD_IO, "call to __kmpc_serialized_parallel\n");

  IncParallelLevel(/*ActiveParallel=*/false);

  if (checkRuntimeUninitialized(loc)) {
    ASSERT0(LT_FUSSY, checkSPMDMode(loc),
            "Expected SPMD mode with uninitialized runtime.");
    return;
  }

  // assume this is only called for nested parallel
  int threadId = GetLogicalThreadIdInBlock(checkSPMDMode(loc));

  // unlike actual parallel, threads in the same team do not share
  // the workTaskDescr in this case and num threads is fixed to 1

  // get current task
  omptarget_nvptx_TaskDescr *currTaskDescr = getMyTopTaskDescriptor(threadId);
  currTaskDescr->SaveLoopData();

  // allocate new task descriptor and copy value from current one, set prev to
  // it
  omptarget_nvptx_TaskDescr *newTaskDescr =
      (omptarget_nvptx_TaskDescr *)SafeMalloc(sizeof(omptarget_nvptx_TaskDescr),
                                              "new seq parallel task");
  newTaskDescr->CopyParent(currTaskDescr);

  // tweak values for serialized parallel case:
  // - each thread becomes ID 0 in its serialized parallel, and
  // - there is only one thread per team
  newTaskDescr->ThreadId() = 0;

  // set new task descriptor as top
  omptarget_nvptx_threadPrivateContext->SetTopLevelTaskDescr(threadId,
                                                             newTaskDescr);
}

EXTERN void __kmpc_end_serialized_parallel(kmp_Ident *loc,
                                           uint32_t global_tid) {
  PRINT0(LD_IO, "call to __kmpc_end_serialized_parallel\n");

  DecParallelLevel(/*ActiveParallel=*/false);

  if (checkRuntimeUninitialized(loc)) {
    ASSERT0(LT_FUSSY, checkSPMDMode(loc),
            "Expected SPMD mode with uninitialized runtime.");
    return;
  }

  // pop stack
  int threadId = GetLogicalThreadIdInBlock(checkSPMDMode(loc));
  omptarget_nvptx_TaskDescr *currTaskDescr = getMyTopTaskDescriptor(threadId);
  // set new top
  omptarget_nvptx_threadPrivateContext->SetTopLevelTaskDescr(
      threadId, currTaskDescr->GetPrevTaskDescr());
  // free
  SafeFree(currTaskDescr, (char *)"new seq parallel task");
  currTaskDescr = getMyTopTaskDescriptor(threadId);
  currTaskDescr->RestoreLoopData();
}

EXTERN uint16_t __kmpc_parallel_level(kmp_Ident *loc, uint32_t global_tid) {
  PRINT0(LD_IO, "call to __kmpc_parallel_level\n");

  return parallelLevel[GetWarpId()] & (OMP_ACTIVE_PARALLEL_LEVEL - 1);
}

// This kmpc call returns the thread id across all teams. It's value is
// cached by the compiler and used when calling the runtime. On nvptx
// it's cheap to recalculate this value so we never use the result
// of this call.
EXTERN int32_t __kmpc_global_thread_num(kmp_Ident *loc) {
  int tid = GetLogicalThreadIdInBlock(checkSPMDMode(loc));
  return GetOmpThreadId(tid, checkSPMDMode(loc));
}

////////////////////////////////////////////////////////////////////////////////
// push params
////////////////////////////////////////////////////////////////////////////////

EXTERN void __kmpc_push_num_threads(kmp_Ident *loc, int32_t tid,
                                    int32_t num_threads) {
  PRINT(LD_IO, "call kmpc_push_num_threads %d\n", num_threads);
  ASSERT0(LT_FUSSY, checkRuntimeInitialized(loc), "Runtime must be initialized.");
  tid = GetLogicalThreadIdInBlock(checkSPMDMode(loc));
  omptarget_nvptx_threadPrivateContext->NumThreadsForNextParallel(tid) =
      num_threads;
}

EXTERN void __kmpc_push_simd_limit(kmp_Ident *loc, int32_t tid,
                                   int32_t simd_limit) {
  PRINT(LD_IO, "call kmpc_push_simd_limit %d\n", (int)simd_limit);
  ASSERT0(LT_FUSSY, checkRuntimeInitialized(loc), "Runtime must be initialized.");
  tid = GetLogicalThreadIdInBlock(checkSPMDMode(loc));
  omptarget_nvptx_threadPrivateContext->SimdLimitForNextSimd(tid) = simd_limit;
}

// Do nothing. The host guarantees we started the requested number of
// teams and we only need inspection of gridDim.

EXTERN void __kmpc_push_num_teams(kmp_Ident *loc, int32_t tid,
                                  int32_t num_teams, int32_t thread_limit) {
  PRINT(LD_IO, "call kmpc_push_num_teams %d\n", (int)num_teams);
  ASSERT0(LT_FUSSY, FALSE,
          "should never have anything with new teams on device");
}

EXTERN void __kmpc_push_proc_bind(kmp_Ident *loc, uint32_t tid,
                                  int proc_bind) {
  PRINT(LD_IO, "call kmpc_push_proc_bind %d\n", (int)proc_bind);
}
