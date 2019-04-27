//===--------- supporti.h - NVPTX OpenMP support functions ------- CUDA -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Wrapper implementation to some functions natively supported by the GPU.
//
//===----------------------------------------------------------------------===//

////////////////////////////////////////////////////////////////////////////////
// Execution Parameters
////////////////////////////////////////////////////////////////////////////////

INLINE void setExecutionParameters(ExecutionMode EMode, RuntimeMode RMode) {
  execution_param = EMode;
  execution_param |= RMode;
}

INLINE bool isGenericMode() { return (execution_param & ModeMask) == Generic; }

INLINE bool isSPMDMode() { return (execution_param & ModeMask) == Spmd; }

INLINE bool isRuntimeUninitialized() {
  return (execution_param & RuntimeMask) == RuntimeUninitialized;
}

INLINE bool isRuntimeInitialized() {
  return (execution_param & RuntimeMask) == RuntimeInitialized;
}

////////////////////////////////////////////////////////////////////////////////
// Execution Modes based on location parameter fields
////////////////////////////////////////////////////////////////////////////////

INLINE bool checkSPMDMode(kmp_Ident *loc) {
  if (!loc)
    return isSPMDMode();

  // If SPMD is true then we are not in the UNDEFINED state so
  // we can return immediately.
  if (loc->reserved_2 & KMP_IDENT_SPMD_MODE)
    return true;

  // If not in SPMD mode and runtime required is a valid
  // combination of flags so we can return immediately.
  if (!(loc->reserved_2 & KMP_IDENT_SIMPLE_RT_MODE))
    return false;

  // We are in underfined state.
  return isSPMDMode();
}

INLINE bool checkGenericMode(kmp_Ident *loc) {
  return !checkSPMDMode(loc);
}

INLINE bool checkRuntimeUninitialized(kmp_Ident *loc) {
  if (!loc)
    return isRuntimeUninitialized();

  // If runtime is required then we know we can't be
  // in the undefined mode. We can return immediately.
  if (!(loc->reserved_2 & KMP_IDENT_SIMPLE_RT_MODE))
    return false;

  // If runtime is required then we need to check is in
  // SPMD mode or not. If not in SPMD mode then we end
  // up in the UNDEFINED state that marks the orphaned
  // functions.
  if (loc->reserved_2 & KMP_IDENT_SPMD_MODE)
    return true;

  // Check if we are in an UNDEFINED state. Undefined is denoted by
  // non-SPMD + noRuntimeRequired which is a combination that
  // cannot actually happen. Undefined states is used to mark orphaned
  // functions.
  return isRuntimeUninitialized();
}

INLINE bool checkRuntimeInitialized(kmp_Ident *loc) {
  return !checkRuntimeUninitialized(loc);
}

////////////////////////////////////////////////////////////////////////////////
// support: get info from machine
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Calls to the NVPTX layer  (assuming 1D layout)
//
////////////////////////////////////////////////////////////////////////////////

INLINE int GetThreadIdInBlock() { return threadIdx.x; }

INLINE int GetBlockIdInKernel() { return blockIdx.x; }

INLINE int GetNumberOfBlocksInKernel() { return gridDim.x; }

INLINE int GetNumberOfThreadsInBlock() { return blockDim.x; }

INLINE unsigned GetWarpId() { return threadIdx.x / WARPSIZE; }

INLINE unsigned GetLaneId() { return threadIdx.x & (WARPSIZE - 1); }

////////////////////////////////////////////////////////////////////////////////
//
// Calls to the Generic Scheme Implementation Layer (assuming 1D layout)
//
////////////////////////////////////////////////////////////////////////////////

// The master thread id is the first thread (lane) of the last warp.
// Thread id is 0 indexed.
// E.g: If NumThreads is 33, master id is 32.
//      If NumThreads is 64, master id is 32.
//      If NumThreads is 97, master id is 96.
//      If NumThreads is 1024, master id is 992.
//
// Called in Generic Execution Mode only.
INLINE int GetMasterThreadID() { return (blockDim.x - 1) & ~(WARPSIZE - 1); }

// The last warp is reserved for the master; other warps are workers.
// Called in Generic Execution Mode only.
INLINE int GetNumberOfWorkersInTeam() { return GetMasterThreadID(); }

////////////////////////////////////////////////////////////////////////////////
// get thread id in team

// This function may be called in a parallel region by the workers
// or a serial region by the master.  If the master (whose CUDA thread
// id is GetMasterThreadID()) calls this routine, we return 0 because
// it is a shadow for the first worker.
INLINE int GetLogicalThreadIdInBlock(bool isSPMDExecutionMode) {
  // Implemented using control flow (predication) instead of with a modulo
  // operation.
  int tid = GetThreadIdInBlock();
  if (!isSPMDExecutionMode && tid >= GetMasterThreadID())
    return 0;
  else
    return tid;
}

////////////////////////////////////////////////////////////////////////////////
//
// OpenMP Thread Support Layer
//
////////////////////////////////////////////////////////////////////////////////

INLINE int GetOmpThreadId(int threadId, bool isSPMDExecutionMode,
                          bool isRuntimeUninitialized) {
  // omp_thread_num
  int rc;

  if (isRuntimeUninitialized) {
    ASSERT0(LT_FUSSY, isSPMDExecutionMode,
            "Uninitialized runtime with non-SPMD mode.");
    // For level 2 parallelism all parallel regions are executed sequentially.
    if (parallelLevel[GetWarpId()] > 0)
      rc = 0;
    else
      rc = GetThreadIdInBlock();
  } else {
    omptarget_nvptx_TaskDescr *currTaskDescr =
        omptarget_nvptx_threadPrivateContext->GetTopLevelTaskDescr(threadId);
    rc = currTaskDescr->ThreadId();
  }
  return rc;
}

INLINE int GetNumberOfOmpThreads(int threadId, bool isSPMDExecutionMode,
                                 bool isRuntimeUninitialized) {
  // omp_num_threads
  int rc;

  if (isRuntimeUninitialized) {
    ASSERT0(LT_FUSSY, isSPMDExecutionMode,
            "Uninitialized runtime with non-SPMD mode.");
    // For level 2 parallelism all parallel regions are executed sequentially.
    if (parallelLevel[GetWarpId()] > 0)
      rc = 1;
    else
      rc = GetNumberOfThreadsInBlock();
  } else {
    omptarget_nvptx_TaskDescr *currTaskDescr =
        omptarget_nvptx_threadPrivateContext->GetTopLevelTaskDescr(threadId);
    ASSERT0(LT_FUSSY, currTaskDescr, "expected a top task descr");
    rc = currTaskDescr->ThreadsInTeam();
  }

  return rc;
}

////////////////////////////////////////////////////////////////////////////////
// Team id linked to OpenMP

INLINE int GetOmpTeamId() {
  // omp_team_num
  return GetBlockIdInKernel(); // assume 1 block per team
}

INLINE int GetNumberOfOmpTeams() {
  // omp_num_teams
  return GetNumberOfBlocksInKernel(); // assume 1 block per team
}

////////////////////////////////////////////////////////////////////////////////
// Masters

INLINE int IsTeamMaster(int ompThreadId) { return (ompThreadId == 0); }

////////////////////////////////////////////////////////////////////////////////
// get OpenMP number of procs

// Get the number of processors in the device.
INLINE int GetNumberOfProcsInDevice(bool isSPMDExecutionMode) {
  if (!isSPMDExecutionMode)
    return GetNumberOfWorkersInTeam();
  return GetNumberOfThreadsInBlock();
}

INLINE int GetNumberOfProcsInTeam(bool isSPMDExecutionMode) {
  return GetNumberOfProcsInDevice(isSPMDExecutionMode);
}

////////////////////////////////////////////////////////////////////////////////
// Memory
////////////////////////////////////////////////////////////////////////////////

INLINE unsigned long PadBytes(unsigned long size,
                              unsigned long alignment) // must be a power of 2
{
  // compute the necessary padding to satisfy alignment constraint
  ASSERT(LT_FUSSY, (alignment & (alignment - 1)) == 0,
         "alignment %lu is not a power of 2\n", alignment);
  return (~(unsigned long)size + 1) & (alignment - 1);
}

INLINE void *SafeMalloc(size_t size, const char *msg) // check if success
{
  void *ptr = malloc(size);
  PRINT(LD_MEM, "malloc data of size %llu for %s: 0x%llx\n",
        (unsigned long long)size, msg, (unsigned long long)ptr);
  return ptr;
}

INLINE void *SafeFree(void *ptr, const char *msg) {
  PRINT(LD_MEM, "free data ptr 0x%llx for %s\n", (unsigned long long)ptr, msg);
  free(ptr);
  return NULL;
}

////////////////////////////////////////////////////////////////////////////////
// Named Barrier Routines
////////////////////////////////////////////////////////////////////////////////

INLINE void named_sync(const int barrier, const int num_threads) {
  asm volatile("bar.sync %0, %1;"
               :
               : "r"(barrier), "r"(num_threads)
               : "memory");
}

////////////////////////////////////////////////////////////////////////////////
// Teams Reduction Scratchpad Helpers
////////////////////////////////////////////////////////////////////////////////

INLINE unsigned int *GetTeamsReductionTimestamp() {
  return static_cast<unsigned int *>(ReductionScratchpadPtr);
}

INLINE char *GetTeamsReductionScratchpad() {
  return static_cast<char *>(ReductionScratchpadPtr) + 256;
}

INLINE void SetTeamsReductionScratchpadPtr(void *ScratchpadPtr) {
  ReductionScratchpadPtr = ScratchpadPtr;
}
