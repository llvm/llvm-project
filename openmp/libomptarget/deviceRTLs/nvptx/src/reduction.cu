//===---- reduction.cu - NVPTX OpenMP reduction implementation ---- CUDA
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of reduction with KMPC interface.
//
//===----------------------------------------------------------------------===//

#include <complex.h>
#include <stdio.h>

#include "omptarget-nvptx.h"

// may eventually remove this
EXTERN
int32_t __gpu_block_reduce() {
  bool isSPMDExecutionMode = isSPMDMode();
  int tid = GetLogicalThreadIdInBlock(isSPMDExecutionMode);
  int nt = GetNumberOfOmpThreads(tid, isSPMDExecutionMode);
  if (nt != blockDim.x)
    return 0;
  unsigned tnum = __ACTIVEMASK();
  if (tnum != (~0x0)) // assume swapSize is 32
    return 0;
  return 1;
}

EXTERN
int32_t __kmpc_reduce_gpu(kmp_Ident *loc, int32_t global_tid, int32_t num_vars,
                          size_t reduce_size, void *reduce_data,
                          void *reduce_array_size, kmp_ReductFctPtr *reductFct,
                          kmp_CriticalName *lck) {
  int threadId = GetLogicalThreadIdInBlock(checkSPMDMode(loc));
  omptarget_nvptx_TaskDescr *currTaskDescr = getMyTopTaskDescriptor(threadId);
  int numthread;
  if (currTaskDescr->IsParallelConstruct()) {
    numthread = GetNumberOfOmpThreads(threadId, checkSPMDMode(loc));
  } else {
    numthread = GetNumberOfOmpTeams();
  }

  if (numthread == 1)
    return 1;
  if (!__gpu_block_reduce())
    return 2;
  if (threadIdx.x == 0)
    return 1;
  return 0;
}

EXTERN
int32_t __kmpc_reduce_combined(kmp_Ident *loc) {
  return threadIdx.x == 0 ? 2 : 0;
}

EXTERN
int32_t __kmpc_reduce_simd(kmp_Ident *loc) {
  return (threadIdx.x % 32 == 0) ? 1 : 0;
}

EXTERN
void __kmpc_nvptx_end_reduce(int32_t global_tid) {}

EXTERN
void __kmpc_nvptx_end_reduce_nowait(int32_t global_tid) {}

EXTERN int32_t __kmpc_shuffle_int32(int32_t val, int16_t delta, int16_t size) {
  return __SHFL_DOWN_SYNC(0xFFFFFFFF, val, delta, size);
}

EXTERN int64_t __kmpc_shuffle_int64(int64_t val, int16_t delta, int16_t size) {
   int lo, hi;
   asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "l"(val));
   hi = __SHFL_DOWN_SYNC(0xFFFFFFFF, hi, delta, size);
   lo = __SHFL_DOWN_SYNC(0xFFFFFFFF, lo, delta, size);
   asm volatile("mov.b64 %0, {%1,%2};" : "=l"(val) : "r"(lo), "r"(hi));
   return val;
}

INLINE static void gpu_regular_warp_reduce(void *reduce_data,
                                           kmp_ShuffleReductFctPtr shflFct) {
  for (uint32_t mask = WARPSIZE / 2; mask > 0; mask /= 2) {
    shflFct(reduce_data, /*LaneId - not used= */ 0,
            /*Offset = */ mask, /*AlgoVersion=*/0);
  }
}

INLINE static void gpu_irregular_warp_reduce(void *reduce_data,
                                             kmp_ShuffleReductFctPtr shflFct,
                                             uint32_t size, uint32_t tid) {
  uint32_t curr_size;
  uint32_t mask;
  curr_size = size;
  mask = curr_size / 2;
  while (mask > 0) {
    shflFct(reduce_data, /*LaneId = */ tid, /*Offset=*/mask, /*AlgoVersion=*/1);
    curr_size = (curr_size + 1) / 2;
    mask = curr_size / 2;
  }
}

INLINE static uint32_t
gpu_irregular_simd_reduce(void *reduce_data, kmp_ShuffleReductFctPtr shflFct) {
  uint32_t lanemask_lt;
  uint32_t lanemask_gt;
  uint32_t size, remote_id, physical_lane_id;
  physical_lane_id = GetThreadIdInBlock() % WARPSIZE;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lanemask_lt));
  uint32_t Liveness = __ACTIVEMASK();
  uint32_t logical_lane_id = __popc(Liveness & lanemask_lt) * 2;
  asm("mov.u32 %0, %%lanemask_gt;" : "=r"(lanemask_gt));
  do {
    Liveness = __ACTIVEMASK();
    remote_id = __ffs(Liveness & lanemask_gt);
    size = __popc(Liveness);
    logical_lane_id /= 2;
    shflFct(reduce_data, /*LaneId =*/logical_lane_id,
            /*Offset=*/remote_id - 1 - physical_lane_id, /*AlgoVersion=*/2);
  } while (logical_lane_id % 2 == 0 && size > 1);
  return (logical_lane_id == 0);
}

EXTERN
int32_t __kmpc_nvptx_simd_reduce_nowait(int32_t global_tid, int32_t num_vars,
                                        size_t reduce_size, void *reduce_data,
                                        kmp_ShuffleReductFctPtr shflFct,
                                        kmp_InterWarpCopyFctPtr cpyFct) {
  uint32_t Liveness = __ACTIVEMASK();
  if (Liveness == 0xffffffff) {
    gpu_regular_warp_reduce(reduce_data, shflFct);
    return GetThreadIdInBlock() % WARPSIZE ==
           0; // Result on lane 0 of the simd warp.
  } else {
    return gpu_irregular_simd_reduce(
        reduce_data, shflFct); // Result on the first active lane.
  }
}

INLINE
static int32_t nvptx_parallel_reduce_nowait(
    int32_t global_tid, int32_t num_vars, size_t reduce_size, void *reduce_data,
    kmp_ShuffleReductFctPtr shflFct, kmp_InterWarpCopyFctPtr cpyFct,
    bool isSPMDExecutionMode, bool isRuntimeUninitialized) {
  uint32_t BlockThreadId = GetLogicalThreadIdInBlock(isSPMDExecutionMode);
  uint32_t NumThreads =
      GetNumberOfOmpThreads(BlockThreadId, isSPMDExecutionMode);
  if (NumThreads == 1)
    return 1;
  /*
   * This reduce function handles reduction within a team. It handles
   * parallel regions in both L1 and L2 parallelism levels. It also
   * supports Generic, SPMD, and NoOMP modes.
   *
   * 1. Reduce within a warp.
   * 2. Warp master copies value to warp 0 via shared memory.
   * 3. Warp 0 reduces to a single value.
   * 4. The reduced value is available in the thread that returns 1.
   */

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  uint32_t WarpsNeeded = (NumThreads + WARPSIZE - 1) / WARPSIZE;
  uint32_t WarpId = BlockThreadId / WARPSIZE;

  // Volta execution model:
  // For the Generic execution mode a parallel region either has 1 thread and
  // beyond that, always a multiple of 32. For the SPMD execution mode we may
  // have any number of threads.
  if ((NumThreads % WARPSIZE == 0) || (WarpId < WarpsNeeded - 1))
    gpu_regular_warp_reduce(reduce_data, shflFct);
  else if (NumThreads > 1) // Only SPMD execution mode comes thru this case.
    gpu_irregular_warp_reduce(reduce_data, shflFct,
                              /*LaneCount=*/NumThreads % WARPSIZE,
                              /*LaneId=*/GetThreadIdInBlock() % WARPSIZE);

  // When we have more than [warpsize] number of threads
  // a block reduction is performed here.
  //
  // Only L1 parallel region can enter this if condition.
  if (NumThreads > WARPSIZE) {
    // Gather all the reduced values from each warp
    // to the first warp.
    cpyFct(reduce_data, WarpsNeeded);

    if (WarpId == 0)
      gpu_irregular_warp_reduce(reduce_data, shflFct, WarpsNeeded,
                                BlockThreadId);
  }
  return BlockThreadId == 0;
#else
  uint32_t Liveness = __ACTIVEMASK();
  if (Liveness == 0xffffffff) // Full warp
    gpu_regular_warp_reduce(reduce_data, shflFct);
  else if (!(Liveness & (Liveness + 1))) // Partial warp but contiguous lanes
    gpu_irregular_warp_reduce(reduce_data, shflFct,
                              /*LaneCount=*/__popc(Liveness),
                              /*LaneId=*/GetThreadIdInBlock() % WARPSIZE);
  else if (!isRuntimeUninitialized) // Dispersed lanes. Only threads in L2
                                    // parallel region may enter here; return
                                    // early.
    return gpu_irregular_simd_reduce(reduce_data, shflFct);

  // When we have more than [warpsize] number of threads
  // a block reduction is performed here.
  //
  // Only L1 parallel region can enter this if condition.
  if (NumThreads > WARPSIZE) {
    uint32_t WarpsNeeded = (NumThreads + WARPSIZE - 1) / WARPSIZE;
    // Gather all the reduced values from each warp
    // to the first warp.
    cpyFct(reduce_data, WarpsNeeded);

    uint32_t WarpId = BlockThreadId / WARPSIZE;
    if (WarpId == 0)
      gpu_irregular_warp_reduce(reduce_data, shflFct, WarpsNeeded,
                                BlockThreadId);

    return BlockThreadId == 0;
  } else if (isRuntimeUninitialized /* Never an L2 parallel region without the OMP runtime */) {
    return BlockThreadId == 0;
  }

  // Get the OMP thread Id. This is different from BlockThreadId in the case of
  // an L2 parallel region.
  return global_tid == 0;
#endif // __CUDA_ARCH__ >= 700
}

EXTERN __attribute__((deprecated)) int32_t __kmpc_nvptx_parallel_reduce_nowait(
    int32_t global_tid, int32_t num_vars, size_t reduce_size, void *reduce_data,
    kmp_ShuffleReductFctPtr shflFct, kmp_InterWarpCopyFctPtr cpyFct) {
  return nvptx_parallel_reduce_nowait(global_tid, num_vars, reduce_size,
                                      reduce_data, shflFct, cpyFct,
                                      isSPMDMode(), isRuntimeUninitialized());
}

EXTERN
int32_t __kmpc_nvptx_parallel_reduce_nowait_v2(
    kmp_Ident *loc, int32_t global_tid, int32_t num_vars, size_t reduce_size,
    void *reduce_data, kmp_ShuffleReductFctPtr shflFct,
    kmp_InterWarpCopyFctPtr cpyFct) {
  return nvptx_parallel_reduce_nowait(
      global_tid, num_vars, reduce_size, reduce_data, shflFct, cpyFct,
      checkSPMDMode(loc), checkRuntimeUninitialized(loc));
}

EXTERN
int32_t __kmpc_nvptx_parallel_reduce_nowait_simple_spmd(
    int32_t global_tid, int32_t num_vars, size_t reduce_size, void *reduce_data,
    kmp_ShuffleReductFctPtr shflFct, kmp_InterWarpCopyFctPtr cpyFct) {
  return nvptx_parallel_reduce_nowait(
      global_tid, num_vars, reduce_size, reduce_data, shflFct, cpyFct,
      /*isSPMDExecutionMode=*/true, /*isRuntimeUninitialized=*/true);
}

EXTERN
int32_t __kmpc_nvptx_parallel_reduce_nowait_simple_generic(
    int32_t global_tid, int32_t num_vars, size_t reduce_size, void *reduce_data,
    kmp_ShuffleReductFctPtr shflFct, kmp_InterWarpCopyFctPtr cpyFct) {
  return nvptx_parallel_reduce_nowait(
      global_tid, num_vars, reduce_size, reduce_data, shflFct, cpyFct,
      /*isSPMDExecutionMode=*/false, /*isRuntimeUninitialized=*/true);
}

INLINE
static int32_t nvptx_teams_reduce_nowait(int32_t global_tid, int32_t num_vars,
                                         size_t reduce_size, void *reduce_data,
                                         kmp_ShuffleReductFctPtr shflFct,
                                         kmp_InterWarpCopyFctPtr cpyFct,
                                         kmp_CopyToScratchpadFctPtr scratchFct,
                                         kmp_LoadReduceFctPtr ldFct,
                                         bool isSPMDExecutionMode) {
  uint32_t ThreadId = GetLogicalThreadIdInBlock(isSPMDExecutionMode);
  // In non-generic mode all workers participate in the teams reduction.
  // In generic mode only the team master participates in the teams
  // reduction because the workers are waiting for parallel work.
  uint32_t NumThreads =
      isSPMDExecutionMode
          ? GetNumberOfOmpThreads(ThreadId, /*isSPMDExecutionMode=*/true)
          : /*Master thread only*/ 1;
  uint32_t TeamId = GetBlockIdInKernel();
  uint32_t NumTeams = GetNumberOfBlocksInKernel();
  __shared__ volatile bool IsLastTeam;

  // Team masters of all teams write to the scratchpad.
  if (ThreadId == 0) {
    unsigned int *timestamp = GetTeamsReductionTimestamp();
    char *scratchpad = GetTeamsReductionScratchpad();

    scratchFct(reduce_data, scratchpad, TeamId, NumTeams);
    __threadfence();

    // atomicInc increments 'timestamp' and has a range [0, NumTeams-1].
    // It resets 'timestamp' back to 0 once the last team increments
    // this counter.
    unsigned val = atomicInc(timestamp, NumTeams - 1);
    IsLastTeam = val == NumTeams - 1;
  }

  // We have to wait on L1 barrier because in GENERIC mode the workers
  // are waiting on barrier 0 for work.
  //
  // If we guard this barrier as follows it leads to deadlock, probably
  // because of a compiler bug: if (!IsGenericMode()) __syncthreads();
  uint16_t SyncWarps = (NumThreads + WARPSIZE - 1) / WARPSIZE;
  named_sync(L1_BARRIER, SyncWarps * WARPSIZE);

  // If this team is not the last, quit.
  if (/* Volatile read by all threads */ !IsLastTeam)
    return 0;

    //
    // Last team processing.
    //

    // Threads in excess of #teams do not participate in reduction of the
    // scratchpad values.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  uint32_t ActiveThreads = NumThreads;
  if (NumTeams < NumThreads) {
    ActiveThreads =
        (NumTeams < WARPSIZE) ? 1 : NumTeams & ~((uint16_t)WARPSIZE - 1);
  }
  if (ThreadId >= ActiveThreads)
    return 0;

  // Load from scratchpad and reduce.
  char *scratchpad = GetTeamsReductionScratchpad();
  ldFct(reduce_data, scratchpad, ThreadId, NumTeams, /*Load only*/ 0);
  for (uint32_t i = ActiveThreads + ThreadId; i < NumTeams; i += ActiveThreads)
    ldFct(reduce_data, scratchpad, i, NumTeams, /*Load and reduce*/ 1);

  uint32_t WarpsNeeded = (ActiveThreads + WARPSIZE - 1) / WARPSIZE;
  uint32_t WarpId = ThreadId / WARPSIZE;

  // Reduce across warps to the warp master.
  if ((ActiveThreads % WARPSIZE == 0) ||
      (WarpId < WarpsNeeded - 1)) // Full warp
    gpu_regular_warp_reduce(reduce_data, shflFct);
  else if (ActiveThreads > 1) // Partial warp but contiguous lanes
    // Only SPMD execution mode comes thru this case.
    gpu_irregular_warp_reduce(reduce_data, shflFct,
                              /*LaneCount=*/ActiveThreads % WARPSIZE,
                              /*LaneId=*/ThreadId % WARPSIZE);

  // When we have more than [warpsize] number of threads
  // a block reduction is performed here.
  if (ActiveThreads > WARPSIZE) {
    // Gather all the reduced values from each warp
    // to the first warp.
    cpyFct(reduce_data, WarpsNeeded);

    if (WarpId == 0)
      gpu_irregular_warp_reduce(reduce_data, shflFct, WarpsNeeded, ThreadId);
  }
#else
  if (ThreadId >= NumTeams)
    return 0;

  // Load from scratchpad and reduce.
  char *scratchpad = GetTeamsReductionScratchpad();
  ldFct(reduce_data, scratchpad, ThreadId, NumTeams, /*Load only*/ 0);
  for (uint32_t i = NumThreads + ThreadId; i < NumTeams; i += NumThreads)
    ldFct(reduce_data, scratchpad, i, NumTeams, /*Load and reduce*/ 1);

  // Reduce across warps to the warp master.
  uint32_t Liveness = __ACTIVEMASK();
  if (Liveness == 0xffffffff) // Full warp
    gpu_regular_warp_reduce(reduce_data, shflFct);
  else // Partial warp but contiguous lanes
    gpu_irregular_warp_reduce(reduce_data, shflFct,
                              /*LaneCount=*/__popc(Liveness),
                              /*LaneId=*/ThreadId % WARPSIZE);

  // When we have more than [warpsize] number of threads
  // a block reduction is performed here.
  uint32_t ActiveThreads = NumTeams < NumThreads ? NumTeams : NumThreads;
  if (ActiveThreads > WARPSIZE) {
    uint32_t WarpsNeeded = (ActiveThreads + WARPSIZE - 1) / WARPSIZE;
    // Gather all the reduced values from each warp
    // to the first warp.
    cpyFct(reduce_data, WarpsNeeded);

    uint32_t WarpId = ThreadId / WARPSIZE;
    if (WarpId == 0)
      gpu_irregular_warp_reduce(reduce_data, shflFct, WarpsNeeded, ThreadId);
  }
#endif // __CUDA_ARCH__ >= 700

  return ThreadId == 0;
}

EXTERN
int32_t __kmpc_nvptx_teams_reduce_nowait(int32_t global_tid, int32_t num_vars,
                                         size_t reduce_size, void *reduce_data,
                                         kmp_ShuffleReductFctPtr shflFct,
                                         kmp_InterWarpCopyFctPtr cpyFct,
                                         kmp_CopyToScratchpadFctPtr scratchFct,
                                         kmp_LoadReduceFctPtr ldFct) {
  return nvptx_teams_reduce_nowait(global_tid, num_vars, reduce_size,
                                   reduce_data, shflFct, cpyFct, scratchFct,
                                   ldFct, isSPMDMode());
}

EXTERN
int32_t __kmpc_nvptx_teams_reduce_nowait_simple_spmd(
    int32_t global_tid, int32_t num_vars, size_t reduce_size, void *reduce_data,
    kmp_ShuffleReductFctPtr shflFct, kmp_InterWarpCopyFctPtr cpyFct,
    kmp_CopyToScratchpadFctPtr scratchFct, kmp_LoadReduceFctPtr ldFct) {
  return nvptx_teams_reduce_nowait(global_tid, num_vars, reduce_size,
                                   reduce_data, shflFct, cpyFct, scratchFct,
                                   ldFct, /*isSPMDExecutionMode=*/true);
}

EXTERN
int32_t __kmpc_nvptx_teams_reduce_nowait_simple_generic(
    int32_t global_tid, int32_t num_vars, size_t reduce_size, void *reduce_data,
    kmp_ShuffleReductFctPtr shflFct, kmp_InterWarpCopyFctPtr cpyFct,
    kmp_CopyToScratchpadFctPtr scratchFct, kmp_LoadReduceFctPtr ldFct) {
  return nvptx_teams_reduce_nowait(global_tid, num_vars, reduce_size,
                                   reduce_data, shflFct, cpyFct, scratchFct,
                                   ldFct, /*isSPMDExecutionMode=*/false);
}

EXTERN int32_t __kmpc_nvptx_teams_reduce_nowait_simple(kmp_Ident *loc,
                                                       int32_t global_tid,
                                                       kmp_CriticalName *crit) {
  if (checkSPMDMode(loc) && GetThreadIdInBlock() != 0)
    return 0;
  // The master thread of the team actually does the reduction.
  while (atomicCAS((uint32_t *)crit, 0, 1))
    ;
  return 1;
}

EXTERN void
__kmpc_nvptx_teams_end_reduce_nowait_simple(kmp_Ident *loc, int32_t global_tid,
                                            kmp_CriticalName *crit) {
  __threadfence_system();
  (void)atomicExch((uint32_t *)crit, 0);
}

INLINE static bool isMaster(kmp_Ident *loc, uint32_t ThreadId) {
  return checkGenericMode(loc) || IsTeamMaster(ThreadId);
}

INLINE static uint32_t roundToWarpsize(uint32_t s) {
  if (s < WARPSIZE)
    return 1;
  return (s & ~(unsigned)(WARPSIZE - 1));
}

__device__ static volatile uint32_t IterCnt = 0;
__device__ static volatile uint32_t Cnt = 0;
EXTERN int32_t __kmpc_nvptx_teams_reduce_nowait_v2(
    kmp_Ident *loc, int32_t global_tid, void *global_buffer,
    int32_t num_of_records, void *reduce_data, kmp_ShuffleReductFctPtr shflFct,
    kmp_InterWarpCopyFctPtr cpyFct, kmp_ListGlobalFctPtr lgcpyFct,
    kmp_ListGlobalFctPtr lgredFct, kmp_ListGlobalFctPtr glcpyFct,
    kmp_ListGlobalFctPtr glredFct) {

  // Terminate all threads in non-SPMD mode except for the master thread.
  if (checkGenericMode(loc) && GetThreadIdInBlock() != GetMasterThreadID())
    return 0;

  uint32_t ThreadId = GetLogicalThreadIdInBlock(checkSPMDMode(loc));

  // In non-generic mode all workers participate in the teams reduction.
  // In generic mode only the team master participates in the teams
  // reduction because the workers are waiting for parallel work.
  uint32_t NumThreads =
      checkSPMDMode(loc)
          ? GetNumberOfOmpThreads(ThreadId, /*isSPMDExecutionMode=*/true)
          : /*Master thread only*/ 1;
  uint32_t TeamId = GetBlockIdInKernel();
  uint32_t NumTeams = GetNumberOfBlocksInKernel();
  __shared__ unsigned Bound;
  __shared__ unsigned ChunkTeamCount;

  // Block progress for teams greater than the current upper
  // limit. We always only allow a number of teams less or equal
  // to the number of slots in the buffer.
  bool IsMaster = isMaster(loc, ThreadId);
  while (IsMaster) {
    // Atomic read
    Bound = atomicAdd((uint32_t *)&IterCnt, 0);
    if (TeamId < Bound + num_of_records)
      break;
  }

  if (IsMaster) {
    int ModBockId = TeamId % num_of_records;
    if (TeamId < num_of_records)
      lgcpyFct(global_buffer, ModBockId, reduce_data);
    else
      lgredFct(global_buffer, ModBockId, reduce_data);
    __threadfence_system();

    // Increment team counter.
    // This counter is incremented by all teams in the current
    // BUFFER_SIZE chunk.
    ChunkTeamCount = atomicInc((uint32_t *)&Cnt, num_of_records - 1);
  }
  // Synchronize
  if (checkSPMDMode(loc))
    __kmpc_barrier(loc, global_tid);

  // reduce_data is global or shared so before being reduced within the
  // warp we need to bring it in local memory:
  // local_reduce_data = reduce_data[i]
  //
  // Example for 3 reduction variables a, b, c (of potentially different
  // types):
  //
  // buffer layout (struct of arrays):
  // a, a, ..., a, b, b, ... b, c, c, ... c
  // |__________|
  //     num_of_records
  //
  // local_data_reduce layout (struct):
  // a, b, c
  //
  // Each thread will have a local struct containing the values to be
  // reduced:
  //      1. do reduction within each warp.
  //      2. do reduction across warps.
  //      3. write the final result to the main reduction variable
  //         by returning 1 in the thread holding the reduction result.

  // Check if this is the very last team.
  unsigned NumRecs = min(NumTeams, num_of_records);
  if (ChunkTeamCount == NumTeams - Bound - 1) {
    //
    // Last team processing.
    //
    if (ThreadId >= NumRecs)
      return 0;
    NumThreads = roundToWarpsize(min(NumThreads, NumRecs));
    if (ThreadId >= NumThreads)
      return 0;

    // Load from buffer and reduce.
    glcpyFct(global_buffer, ThreadId, reduce_data);
    for (uint32_t i = NumThreads + ThreadId; i < NumRecs; i += NumThreads)
      glredFct(global_buffer, i, reduce_data);

    // Reduce across warps to the warp master.
    if (NumThreads > 1) {
      gpu_regular_warp_reduce(reduce_data, shflFct);

      // When we have more than [warpsize] number of threads
      // a block reduction is performed here.
      uint32_t ActiveThreads = min(NumRecs, NumThreads);
      if (ActiveThreads > WARPSIZE) {
        uint32_t WarpsNeeded = (ActiveThreads + WARPSIZE - 1) / WARPSIZE;
        // Gather all the reduced values from each warp
        // to the first warp.
        cpyFct(reduce_data, WarpsNeeded);

        uint32_t WarpId = ThreadId / WARPSIZE;
        if (WarpId == 0)
          gpu_irregular_warp_reduce(reduce_data, shflFct, WarpsNeeded,
                                    ThreadId);
      }
    }

    if (IsMaster) {
      Cnt = 0;
      IterCnt = 0;
      return 1;
    }
    return 0;
  }
  if (IsMaster && ChunkTeamCount == num_of_records - 1) {
    // Allow SIZE number of teams to proceed writing their
    // intermediate results to the global buffer.
    atomicAdd((uint32_t *)&IterCnt, num_of_records);
  }

  return 0;
}

