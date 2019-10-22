//===--------- support.h - NVPTX OpenMP support functions -------- CUDA -*-===//
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

////////////////////////////////////////////////////////////////////////////////
// Execution Parameters
////////////////////////////////////////////////////////////////////////////////
enum ExecutionMode {
  Generic = 0x00u,
  Spmd = 0x01u,
  ModeMask = 0x01u,
};

enum RuntimeMode {
  RuntimeInitialized = 0x00u,
  RuntimeUninitialized = 0x02u,
  RuntimeMask = 0x02u,
};

INLINE void setExecutionParameters(ExecutionMode EMode, RuntimeMode RMode);
INLINE bool isGenericMode();
INLINE bool isSPMDMode();
INLINE bool isRuntimeUninitialized();
INLINE bool isRuntimeInitialized();

////////////////////////////////////////////////////////////////////////////////
// get info from machine
////////////////////////////////////////////////////////////////////////////////

// get low level ids of resources
INLINE int GetThreadIdInBlock();
INLINE int GetBlockIdInKernel();
INLINE int GetNumberOfBlocksInKernel();
INLINE int GetNumberOfThreadsInBlock();
INLINE unsigned GetWarpId();
INLINE unsigned GetLaneId();

// get global ids to locate tread/team info (constant regardless of OMP)
INLINE int GetLogicalThreadIdInBlock(bool isSPMDExecutionMode);
INLINE int GetMasterThreadID();
INLINE int GetNumberOfWorkersInTeam();

// get OpenMP thread and team ids
INLINE int GetOmpThreadId(int threadId,
                          bool isSPMDExecutionMode);    // omp_thread_num
INLINE int GetOmpTeamId();                              // omp_team_num

// get OpenMP number of threads and team
INLINE int GetNumberOfOmpThreads(bool isSPMDExecutionMode); // omp_num_threads
INLINE int GetNumberOfOmpTeams();                           // omp_num_teams

// get OpenMP number of procs
INLINE int GetNumberOfProcsInTeam(bool isSPMDExecutionMode);
INLINE int GetNumberOfProcsInDevice(bool isSPMDExecutionMode);

// masters
INLINE int IsTeamMaster(int ompThreadId);

// Parallel level
INLINE void IncParallelLevel(bool ActiveParallel);
INLINE void DecParallelLevel(bool ActiveParallel);

////////////////////////////////////////////////////////////////////////////////
// Memory
////////////////////////////////////////////////////////////////////////////////

// safe alloc and free
INLINE void *SafeMalloc(size_t size, const char *msg); // check if success
INLINE void *SafeFree(void *ptr, const char *msg);
// pad to a alignment (power of 2 only)
INLINE unsigned long PadBytes(unsigned long size, unsigned long alignment);
#define ADD_BYTES(_addr, _bytes)                                               \
  ((void *)((char *)((void *)(_addr)) + (_bytes)))
#define SUB_BYTES(_addr, _bytes)                                               \
  ((void *)((char *)((void *)(_addr)) - (_bytes)))

////////////////////////////////////////////////////////////////////////////////
// Named Barrier Routines
////////////////////////////////////////////////////////////////////////////////
INLINE void named_sync(const int barrier, const int num_threads);

////////////////////////////////////////////////////////////////////////////////
// Teams Reduction Scratchpad Helpers
////////////////////////////////////////////////////////////////////////////////
INLINE unsigned int *GetTeamsReductionTimestamp();
INLINE char *GetTeamsReductionScratchpad();
INLINE void SetTeamsReductionScratchpadPtr(void *ScratchpadPtr);
