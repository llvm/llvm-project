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

void setExecutionParameters(ExecutionMode EMode, RuntimeMode RMode);
bool isGenericMode();
bool isRuntimeUninitialized();
bool isRuntimeInitialized();

////////////////////////////////////////////////////////////////////////////////
// get info from machine
////////////////////////////////////////////////////////////////////////////////

// get global ids to locate tread/team info (constant regardless of OMP)
int GetLogicalThreadIdInBlock();
int GetMasterThreadID();
int GetNumberOfWorkersInTeam();

// get OpenMP thread and team ids
int GetOmpThreadId();                         // omp_thread_num
int GetOmpTeamId();                           // omp_team_num

// get OpenMP number of threads and team
int GetNumberOfOmpThreads(bool isSPMDExecutionMode); // omp_num_threads
int GetNumberOfOmpTeams();                           // omp_num_teams

// get OpenMP number of procs
int GetNumberOfProcsInTeam(bool isSPMDExecutionMode);
int GetNumberOfProcsInDevice(bool isSPMDExecutionMode);

// masters
int IsTeamMaster(int ompThreadId);

// Parallel level
void IncParallelLevel(bool ActiveParallel, __kmpc_impl_lanemask_t Mask);
void DecParallelLevel(bool ActiveParallel, __kmpc_impl_lanemask_t Mask);

////////////////////////////////////////////////////////////////////////////////
// Memory
////////////////////////////////////////////////////////////////////////////////

// safe alloc and free
void *SafeMalloc(size_t size, const char *msg); // check if success
void *SafeFree(void *ptr, const char *msg);
// pad to a alignment (power of 2 only)
unsigned long PadBytes(unsigned long size, unsigned long alignment);
#define ADD_BYTES(_addr, _bytes)                                               \
  ((void *)((char *)((void *)(_addr)) + (_bytes)))
#define SUB_BYTES(_addr, _bytes)                                               \
  ((void *)((char *)((void *)(_addr)) - (_bytes)))

////////////////////////////////////////////////////////////////////////////////
// Teams Reduction Scratchpad Helpers
////////////////////////////////////////////////////////////////////////////////
unsigned int *GetTeamsReductionTimestamp();
char *GetTeamsReductionScratchpad();

// Invoke an outlined parallel function unwrapping global, shared arguments (up
// to 128).
void __kmp_invoke_microtask(kmp_int32 global_tid, kmp_int32 bound_tid, void *fn,
                            void **args, size_t nargs);

#endif
