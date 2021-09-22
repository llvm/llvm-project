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
#pragma omp declare target

#include "common/debug.h"
#include "common/omptarget.h"
#include "common/support.h"

////////////////////////////////////////////////////////////////////////////////
// Execution Parameters
////////////////////////////////////////////////////////////////////////////////

void setExecutionParameters(OMPTgtExecModeFlags EMode,
                            OMPTgtRuntimeModeFlags RMode) {
  execution_param = EMode;
  execution_param |= RMode;
}

bool isGenericMode() { return execution_param & OMP_TGT_EXEC_MODE_GENERIC; }

bool isRuntimeUninitialized() { return !isRuntimeInitialized(); }

bool isRuntimeInitialized() {
  return execution_param & OMP_TGT_RUNTIME_INITIALIZED;
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
int GetMasterThreadID() {
  return (__kmpc_get_hardware_num_threads_in_block() - 1) & ~(WARPSIZE - 1);
}

// The last warp is reserved for the master; other warps are workers.
// Called in Generic Execution Mode only.
int GetNumberOfWorkersInTeam() { return GetMasterThreadID(); }

////////////////////////////////////////////////////////////////////////////////
// get thread id in team

// This function may be called in a parallel region by the workers
// or a serial region by the master.  If the master (whose CUDA thread
// id is GetMasterThreadID()) calls this routine, we return 0 because
// it is a shadow for the first worker.
int GetLogicalThreadIdInBlock() {
  // Implemented using control flow (predication) instead of with a modulo
  // operation.
  int tid = __kmpc_get_hardware_thread_id_in_block();
  if (__kmpc_is_generic_main_thread(tid))
    return 0;
  else
    return tid;
}

////////////////////////////////////////////////////////////////////////////////
//
// OpenMP Thread Support Layer
//
////////////////////////////////////////////////////////////////////////////////

int GetOmpThreadId() {
  int tid = __kmpc_get_hardware_thread_id_in_block();
  if (__kmpc_is_generic_main_thread(tid))
    return 0;
  // omp_thread_num
  int rc;
  if (__kmpc_parallel_level() > 1) {
    rc = 0;
  } else if (__kmpc_is_spmd_exec_mode()) {
    rc = tid;
  } else {
    omptarget_nvptx_TaskDescr *currTaskDescr =
        omptarget_nvptx_threadPrivateContext->GetTopLevelTaskDescr(tid);
    ASSERT0(LT_FUSSY, currTaskDescr, "expected a top task descr");
    rc = currTaskDescr->ThreadId();
  }
  return rc;
}

int GetNumberOfOmpThreads(bool isSPMDExecutionMode) {
  // omp_num_threads
  int rc;
  int Level = parallelLevel[GetWarpId()];
  if (Level != OMP_ACTIVE_PARALLEL_LEVEL + 1) {
    rc = 1;
  } else if (isSPMDExecutionMode) {
    rc = __kmpc_get_hardware_num_threads_in_block();
  } else {
    rc = threadsInTeam;
  }

  return rc;
}

////////////////////////////////////////////////////////////////////////////////
// Team id linked to OpenMP

int GetOmpTeamId() {
  // omp_team_num
  return GetBlockIdInKernel(); // assume 1 block per team
}

int GetNumberOfOmpTeams() {
  // omp_num_teams
  return __kmpc_get_hardware_num_blocks(); // assume 1 block per team
}

////////////////////////////////////////////////////////////////////////////////
// Masters

int IsTeamMaster(int ompThreadId) { return (ompThreadId == 0); }

////////////////////////////////////////////////////////////////////////////////
// Parallel level

void IncParallelLevel(bool ActiveParallel, __kmpc_impl_lanemask_t Mask) {
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

void DecParallelLevel(bool ActiveParallel, __kmpc_impl_lanemask_t Mask) {
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
int GetNumberOfProcsInDevice(bool isSPMDExecutionMode) {
  if (!isSPMDExecutionMode)
    return GetNumberOfWorkersInTeam();
  return __kmpc_get_hardware_num_threads_in_block();
}

int GetNumberOfProcsInTeam(bool isSPMDExecutionMode) {
  return GetNumberOfProcsInDevice(isSPMDExecutionMode);
}

////////////////////////////////////////////////////////////////////////////////
// Memory
////////////////////////////////////////////////////////////////////////////////

unsigned long PadBytes(unsigned long size,
                       unsigned long alignment) // must be a power of 2
{
  // compute the necessary padding to satisfy alignment constraint
  ASSERT(LT_FUSSY, (alignment & (alignment - 1)) == 0,
         "alignment %lu is not a power of 2\n", alignment);
  return (~(unsigned long)size + 1) & (alignment - 1);
}

void *SafeMalloc(size_t size, const char *msg) // check if success
{
  void *ptr = __kmpc_impl_malloc(size);
  PRINT(LD_MEM, "malloc data of size %llu for %s: 0x%llx\n",
        (unsigned long long)size, msg, (unsigned long long)ptr);
  return ptr;
}

void *SafeFree(void *ptr, const char *msg) {
  PRINT(LD_MEM, "free data ptr 0x%llx for %s\n", (unsigned long long)ptr, msg);
  __kmpc_impl_free(ptr);
  return NULL;
}

////////////////////////////////////////////////////////////////////////////////
// Teams Reduction Scratchpad Helpers
////////////////////////////////////////////////////////////////////////////////

unsigned int *GetTeamsReductionTimestamp() {
  return static_cast<unsigned int *>(ReductionScratchpadPtr);
}

char *GetTeamsReductionScratchpad() {
  return static_cast<char *>(ReductionScratchpadPtr) + 256;
}

// Invoke an outlined parallel function unwrapping arguments (up
// to 32).
void __kmp_invoke_microtask(kmp_int32 global_tid, kmp_int32 bound_tid, void *fn,
                            void **args, size_t nargs) {
  switch (nargs) {
#include "common/generated_microtask_cases.gen"
  default:
    printf("Too many arguments in kmp_invoke_microtask, aborting execution.\n");
    __builtin_trap();
  }
}

#pragma omp end declare target
