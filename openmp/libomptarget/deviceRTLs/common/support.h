//===--------- support.h - OpenMP GPU support functions ---------- CUDA -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Wrapper to some functions natively supported by the GPU.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_SUPPORT_H
#define OMPTARGET_SUPPORT_H

#include "interface.h"
#include "target_impl.h"

////////////////////////////////////////////////////////////////////////////////
// Execution Parameters
////////////////////////////////////////////////////////////////////////////////
enum ExecutionMode {
  Spmd = 0x00u,
  Generic = 0x01u,
  ModeMask = 0x01u,
};

enum RuntimeMode {
  RuntimeInitialized = 0x00u,
  RuntimeUninitialized = 0x02u,
  RuntimeMask = 0x02u,
};

DEVICE void setExecutionParameters(ExecutionMode EMode, RuntimeMode RMode);
DEVICE bool isGenericMode();
DEVICE bool isSPMDMode();
DEVICE bool isRuntimeUninitialized();
DEVICE bool isRuntimeInitialized();

////////////////////////////////////////////////////////////////////////////////
// Execution Modes based on location parameter fields
////////////////////////////////////////////////////////////////////////////////

DEVICE bool checkSPMDMode(kmp_Ident *loc);
DEVICE bool checkGenericMode(kmp_Ident *loc);
DEVICE bool checkRuntimeUninitialized(kmp_Ident *loc);
DEVICE bool checkRuntimeInitialized(kmp_Ident *loc);

////////////////////////////////////////////////////////////////////////////////
// get info from machine
////////////////////////////////////////////////////////////////////////////////

// get global ids to locate tread/team info (constant regardless of OMP)
DEVICE int GetLogicalThreadIdInBlock(bool isSPMDExecutionMode);
DEVICE int GetMasterThreadID();
DEVICE int GetNumberOfWorkersInTeam();

// get OpenMP thread and team ids
DEVICE int GetOmpThreadId(int threadId,
                          bool isSPMDExecutionMode); // omp_thread_num
DEVICE int GetOmpTeamId();                           // omp_team_num

// get OpenMP number of threads and team
DEVICE int GetNumberOfOmpThreads(bool isSPMDExecutionMode); // omp_num_threads
DEVICE int GetNumberOfOmpTeams();                           // omp_num_teams

// get OpenMP number of procs
DEVICE int GetNumberOfProcsInTeam(bool isSPMDExecutionMode);
DEVICE int GetNumberOfProcsInDevice(bool isSPMDExecutionMode);

// masters
DEVICE int IsTeamMaster(int ompThreadId);

// Parallel level
DEVICE void IncParallelLevel(bool ActiveParallel, __kmpc_impl_lanemask_t Mask);
DEVICE void DecParallelLevel(bool ActiveParallel, __kmpc_impl_lanemask_t Mask);

////////////////////////////////////////////////////////////////////////////////
// Memory
////////////////////////////////////////////////////////////////////////////////

// safe alloc and free
DEVICE void *SafeMalloc(size_t size, const char *msg); // check if success
DEVICE void *SafeFree(void *ptr, const char *msg);
// pad to a alignment (power of 2 only)
DEVICE unsigned long PadBytes(unsigned long size, unsigned long alignment);
#define ADD_BYTES(_addr, _bytes)                                               \
  ((void *)((char *)((void *)(_addr)) + (_bytes)))
#define SUB_BYTES(_addr, _bytes)                                               \
  ((void *)((char *)((void *)(_addr)) - (_bytes)))

////////////////////////////////////////////////////////////////////////////////
// Teams Reduction Scratchpad Helpers
////////////////////////////////////////////////////////////////////////////////
DEVICE unsigned int *GetTeamsReductionTimestamp();
DEVICE char *GetTeamsReductionScratchpad();

#endif
