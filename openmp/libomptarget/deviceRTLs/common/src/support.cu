//===--------- support.cu - GPU OpenMP support functions --------- CUDA -*-===//
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

#include "common/support.h"
#include "common/debug.h"
#include "common/omptarget.h"

////////////////////////////////////////////////////////////////////////////////
// Execution Parameters
////////////////////////////////////////////////////////////////////////////////

DEVICE void setExecutionParameters(ExecutionMode EMode, RuntimeMode RMode) {
  execution_param = EMode;
  execution_param |= RMode;
}

DEVICE bool isGenericMode() { return (execution_param & ModeMask) == Generic; }

DEVICE bool isSPMDMode() { return (execution_param & ModeMask) == Spmd; }

DEVICE bool isRuntimeUninitialized() {
  return (execution_param & RuntimeMask) == RuntimeUninitialized;
}

DEVICE bool isRuntimeInitialized() {
  return (execution_param & RuntimeMask) == RuntimeInitialized;
}

////////////////////////////////////////////////////////////////////////////////
// Execution Modes based on location parameter fields
////////////////////////////////////////////////////////////////////////////////

DEVICE bool checkSPMDMode(kmp_Ident *loc) {
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

DEVICE bool checkGenericMode(kmp_Ident *loc) {
  return !checkSPMDMode(loc);
}

DEVICE bool checkRuntimeUninitialized(kmp_Ident *loc) {
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

DEVICE bool checkRuntimeInitialized(kmp_Ident *loc) {
  return !checkRuntimeUninitialized(loc);
}

////////////////////////////////////////////////////////////////////////////////
// support: get info from machine
////////////////////////////////////////////////////////////////////////////////

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
DEVICE int GetMasterThreadID() { return (GetNumberOfThreadsInBlock() - 1) & ~(WARPSIZE - 1); }

// The last warp is reserved for the master; other warps are workers.
// Called in Generic Execution Mode only.
DEVICE int GetNumberOfWorkersInTeam() { return GetMasterThreadID(); }

////////////////////////////////////////////////////////////////////////////////
// get thread id in team

// This function may be called in a parallel region by the workers
// or a serial region by the master.  If the master (whose CUDA thread
// id is GetMasterThreadID()) calls this routine, we return 0 because
// it is a shadow for the first worker.
DEVICE int GetLogicalThreadIdInBlock(bool isSPMDExecutionMode) {
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

DEVICE int GetOmpThreadId(int threadId, bool isSPMDExecutionMode) {
  // omp_thread_num
  int rc;
  if ((parallelLevel[GetWarpId()] & (OMP_ACTIVE_PARALLEL_LEVEL - 1)) > 1) {
    rc = 0;
  } else if (isSPMDExecutionMode) {
    rc = GetThreadIdInBlock();
  } else {
    omptarget_nvptx_TaskDescr *currTaskDescr =
        omptarget_nvptx_threadPrivateContext->GetTopLevelTaskDescr(threadId);
    ASSERT0(LT_FUSSY, currTaskDescr, "expected a top task descr");
    rc = currTaskDescr->ThreadId();
  }
  return rc;
}

DEVICE int GetNumberOfOmpThreads(bool isSPMDExecutionMode) {
  // omp_num_threads
  int rc;
  int Level = parallelLevel[GetWarpId()];
  if (Level != OMP_ACTIVE_PARALLEL_LEVEL + 1) {
    rc = 1;
  } else if (isSPMDExecutionMode) {
    rc = GetNumberOfThreadsInBlock();
  } else {
    rc = threadsInTeam;
  }

  return rc;
}

////////////////////////////////////////////////////////////////////////////////
// Team id linked to OpenMP

DEVICE int GetOmpTeamId() {
  // omp_team_num
  return GetBlockIdInKernel(); // assume 1 block per team
}

DEVICE int GetNumberOfOmpTeams() {
  // omp_num_teams
  return GetNumberOfBlocksInKernel(); // assume 1 block per team
}

////////////////////////////////////////////////////////////////////////////////
// Masters

DEVICE int IsTeamMaster(int ompThreadId) { return (ompThreadId == 0); }

////////////////////////////////////////////////////////////////////////////////
// Parallel level

DEVICE void IncParallelLevel(bool ActiveParallel, __kmpc_impl_lanemask_t Mask) {
  __kmpc_impl_syncwarp(Mask);
  __kmpc_impl_lanemask_t LaneMaskLt = __kmpc_impl_lanemask_lt();
  unsigned Rank = __kmpc_impl_popc(Mask & LaneMaskLt);
  if (Rank == 0) {
    parallelLevel[GetWarpId()] +=
        (1 + (ActiveParallel ? OMP_ACTIVE_PARALLEL_LEVEL : 0));
    __kmpc_impl_threadfence();
  }
  __kmpc_impl_syncwarp(Mask);
}

DEVICE void DecParallelLevel(bool ActiveParallel, __kmpc_impl_lanemask_t Mask) {
  __kmpc_impl_syncwarp(Mask);
  __kmpc_impl_lanemask_t LaneMaskLt = __kmpc_impl_lanemask_lt();
  unsigned Rank = __kmpc_impl_popc(Mask & LaneMaskLt);
  if (Rank == 0) {
    parallelLevel[GetWarpId()] -=
        (1 + (ActiveParallel ? OMP_ACTIVE_PARALLEL_LEVEL : 0));
    __kmpc_impl_threadfence();
  }
  __kmpc_impl_syncwarp(Mask);
}

////////////////////////////////////////////////////////////////////////////////
// get OpenMP number of procs

// Get the number of processors in the device.
DEVICE int GetNumberOfProcsInDevice(bool isSPMDExecutionMode) {
  if (!isSPMDExecutionMode)
    return GetNumberOfWorkersInTeam();
  return GetNumberOfThreadsInBlock();
}

DEVICE int GetNumberOfProcsInTeam(bool isSPMDExecutionMode) {
  return GetNumberOfProcsInDevice(isSPMDExecutionMode);
}

////////////////////////////////////////////////////////////////////////////////
// Memory
////////////////////////////////////////////////////////////////////////////////

DEVICE unsigned long PadBytes(unsigned long size,
                              unsigned long alignment) // must be a power of 2
{
  // compute the necessary padding to satisfy alignment constraint
  ASSERT(LT_FUSSY, (alignment & (alignment - 1)) == 0,
         "alignment %lu is not a power of 2\n", alignment);
  return (~(unsigned long)size + 1) & (alignment - 1);
}

DEVICE void *SafeMalloc(size_t size, const char *msg) // check if success
{
  void *ptr = __kmpc_impl_malloc(size);
  PRINT(LD_MEM, "malloc data of size %llu for %s: 0x%llx\n",
        (unsigned long long)size, msg, (unsigned long long)ptr);
  return ptr;
}

DEVICE void *SafeFree(void *ptr, const char *msg) {
  PRINT(LD_MEM, "free data ptr 0x%llx for %s\n", (unsigned long long)ptr, msg);
  __kmpc_impl_free(ptr);
  return NULL;
}

////////////////////////////////////////////////////////////////////////////////
// Teams Reduction Scratchpad Helpers
////////////////////////////////////////////////////////////////////////////////

DEVICE unsigned int *GetTeamsReductionTimestamp() {
  return static_cast<unsigned int *>(ReductionScratchpadPtr);
}

DEVICE char *GetTeamsReductionScratchpad() {
  return static_cast<char *>(ReductionScratchpadPtr) + 256;
}

