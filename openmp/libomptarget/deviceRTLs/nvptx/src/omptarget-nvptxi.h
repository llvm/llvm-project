//===---- omptarget-nvptxi.h - NVPTX OpenMP GPU initialization --- CUDA -*-===//
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

////////////////////////////////////////////////////////////////////////////////
// Task Descriptor
////////////////////////////////////////////////////////////////////////////////

INLINE omp_sched_t omptarget_nvptx_TaskDescr::GetRuntimeSched() {
  // sched starts from 1..4; encode it as 0..3; so add 1 here
  uint8_t rc = (items.flags & TaskDescr_SchedMask) + 1;
  return (omp_sched_t)rc;
}

INLINE void omptarget_nvptx_TaskDescr::SetRuntimeSched(omp_sched_t sched) {
  // sched starts from 1..4; encode it as 0..3; so sub 1 here
  uint8_t val = ((uint8_t)sched) - 1;
  // clear current sched
  items.flags &= ~TaskDescr_SchedMask;
  // set new sched
  items.flags |= val;
}

INLINE void omptarget_nvptx_TaskDescr::InitLevelZeroTaskDescr() {
  // slow method
  // flag:
  //   default sched is static,
  //   dyn is off (unused now anyway, but may need to sample from host ?)
  //   not in parallel

  items.flags = 0;
  items.nthreads = GetNumberOfProcsInTeam();
  ;                                // threads: whatever was alloc by kernel
  items.threadId = 0;         // is master
  items.threadsInTeam = 1;    // sequential
  items.runtimeChunkSize = 1; // prefered chunking statik with chunk 1
}

// This is called when all threads are started together in SPMD mode.
// OMP directives include target parallel, target distribute parallel for, etc.
INLINE void omptarget_nvptx_TaskDescr::InitLevelOneTaskDescr(
    uint16_t tnum, omptarget_nvptx_TaskDescr *parentTaskDescr) {
  // slow method
  // flag:
  //   default sched is static,
  //   dyn is off (unused now anyway, but may need to sample from host ?)
  //   in L1 parallel

  items.flags =
      TaskDescr_InPar | TaskDescr_IsParConstr; // set flag to parallel
  items.nthreads = 0; // # threads for subsequent parallel region
  items.threadId =
      GetThreadIdInBlock(); // get ids from cuda (only called for 1st level)
  items.threadsInTeam = tnum;
  items.runtimeChunkSize = 1; // prefered chunking statik with chunk 1
  prev = parentTaskDescr;
}

INLINE void omptarget_nvptx_TaskDescr::CopyData(
    omptarget_nvptx_TaskDescr *sourceTaskDescr) {
  items = sourceTaskDescr->items;
}

INLINE void
omptarget_nvptx_TaskDescr::Copy(omptarget_nvptx_TaskDescr *sourceTaskDescr) {
  CopyData(sourceTaskDescr);
  prev = sourceTaskDescr->prev;
}

INLINE void omptarget_nvptx_TaskDescr::CopyParent(
    omptarget_nvptx_TaskDescr *parentTaskDescr) {
  CopyData(parentTaskDescr);
  prev = parentTaskDescr;
}

INLINE void omptarget_nvptx_TaskDescr::CopyForExplicitTask(
    omptarget_nvptx_TaskDescr *parentTaskDescr) {
  CopyParent(parentTaskDescr);
  items.flags = items.flags & ~TaskDescr_IsParConstr;
  ASSERT0(LT_FUSSY, IsTaskConstruct(), "expected task");
}

INLINE void omptarget_nvptx_TaskDescr::CopyToWorkDescr(
    omptarget_nvptx_TaskDescr *masterTaskDescr, uint16_t tnum) {
  CopyParent(masterTaskDescr);
  // overrwrite specific items;
  items.flags |=
      TaskDescr_InPar | TaskDescr_IsParConstr; // set flag to parallel
  items.threadsInTeam = tnum;             // set number of threads
}

INLINE void omptarget_nvptx_TaskDescr::CopyFromWorkDescr(
    omptarget_nvptx_TaskDescr *workTaskDescr) {
  Copy(workTaskDescr);
  //
  // overrwrite specific items;
  //
  // The threadID should be GetThreadIdInBlock() % GetMasterThreadID().
  // This is so that the serial master (first lane in the master warp)
  // gets a threadId of 0.
  // However, we know that this function is always called in a parallel
  // region where only workers are active.  The serial master thread
  // never enters this region.  When a parallel region is executed serially,
  // the threadId is set to 0 elsewhere and the kmpc_serialized_* functions
  // are called, which never activate this region.
  items.threadId =
      GetThreadIdInBlock(); // get ids from cuda (only called for 1st level)
}

INLINE void omptarget_nvptx_TaskDescr::CopyConvergentParent(
    omptarget_nvptx_TaskDescr *parentTaskDescr, uint16_t tid, uint16_t tnum) {
  CopyParent(parentTaskDescr);
  items.flags |= TaskDescr_InParL2P; // In L2+ parallelism
  items.threadsInTeam = tnum;        // set number of threads
  items.threadId = tid;
}

////////////////////////////////////////////////////////////////////////////////
// Thread Private Context
////////////////////////////////////////////////////////////////////////////////

INLINE omptarget_nvptx_TaskDescr *
omptarget_nvptx_ThreadPrivateContext::GetTopLevelTaskDescr(int tid) {
  ASSERT0(
      LT_FUSSY, tid < MAX_THREADS_PER_TEAM,
      "Getting top level, tid is larger than allocated data structure size");
  return topTaskDescr[tid];
}

INLINE void
omptarget_nvptx_ThreadPrivateContext::InitThreadPrivateContext(int tid) {
  // levelOneTaskDescr is init when starting the parallel region
  // top task descr is NULL (team master version will be fixed separately)
  topTaskDescr[tid] = NULL;
  // no num threads value has been pushed
  nextRegion.tnum[tid] = 0;
  // priv counter init to zero
  priv[tid] = 0;
  // the following don't need to be init here; they are init when using dyn
  // sched
  // current_Event, events_Number, chunk, num_Iterations, schedule
}

////////////////////////////////////////////////////////////////////////////////
// Work Descriptor
////////////////////////////////////////////////////////////////////////////////

INLINE void omptarget_nvptx_WorkDescr::InitWorkDescr() {
  cg.Clear(); // start and stop to zero too
  // threadsInParallelTeam does not need to be init (done in start parallel)
  hasCancel = FALSE;
}

////////////////////////////////////////////////////////////////////////////////
// Team Descriptor
////////////////////////////////////////////////////////////////////////////////

INLINE void omptarget_nvptx_TeamDescr::InitTeamDescr() {
  levelZeroTaskDescr.InitLevelZeroTaskDescr();
  workDescrForActiveParallel.InitWorkDescr();
  // omp_init_lock(criticalLock);
}

////////////////////////////////////////////////////////////////////////////////
// Get private data structure for thread
////////////////////////////////////////////////////////////////////////////////

// Utility routines for CUDA threads
INLINE omptarget_nvptx_TeamDescr &getMyTeamDescriptor() {
  return omptarget_nvptx_threadPrivateContext->TeamContext();
}

INLINE omptarget_nvptx_WorkDescr &getMyWorkDescriptor() {
  omptarget_nvptx_TeamDescr &currTeamDescr = getMyTeamDescriptor();
  return currTeamDescr.WorkDescr();
}

INLINE omptarget_nvptx_TaskDescr *getMyTopTaskDescriptor(int threadId) {
  return omptarget_nvptx_threadPrivateContext->GetTopLevelTaskDescr(threadId);
}

INLINE omptarget_nvptx_TaskDescr *getMyTopTaskDescriptor() {
  return getMyTopTaskDescriptor(GetLogicalThreadIdInBlock());
}
