//===---- Reduction.cpp - OpenMP device reduction implementation - C++ -*-===//
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

#include "Debug.h"
#include "Interface.h"
#include "Mapping.h"
#include "State.h"
#include "Synchronization.h"
#include "Types.h"
#include "Utils.h"

using namespace ompx;

namespace {

#pragma omp begin declare target device_type(nohost)

void gpu_regular_warp_reduce(void *reduce_data, ShuffleReductFnTy shflFct) {
  for (uint32_t mask = mapping::getWarpSize() / 2; mask > 0; mask /= 2) {
    shflFct(reduce_data, /*LaneId - not used= */ 0,
            /*Offset = */ mask, /*AlgoVersion=*/0);
  }
}

void gpu_irregular_warp_reduce(void *reduce_data, ShuffleReductFnTy shflFct,
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

static int32_t nvptx_parallel_reduce_nowait(void *reduce_data,
                                            ShuffleReductFnTy shflFct,
                                            InterWarpCopyFnTy cpyFct) {
  uint32_t NumThreads = omp_get_num_threads();
  // Handle degenerated parallel regions, including all nested ones, first.
  if (NumThreads == 1)
    return 1;

  /*
   * 1. Reduce within a warp.
   * 2. Warp master copies value to warp 0 via shared memory.
   * 3. Warp 0 reduces to a single value.
   * 4. The reduced value is available in the thread that returns 1.
   */

  uint32_t BlockThreadId = mapping::getThreadIdInBlock();
  uint32_t NumWarps =
      (NumThreads + mapping::getWarpSize() - 1) / mapping::getWarpSize();

  // For the Generic execution mode a parallel region either has 1 thread and
  // beyond that, always a multiple of 32. For the SPMD execution mode we may
  // have any number of threads.
  gpu_regular_warp_reduce(reduce_data, shflFct);

  // When we have more than [mapping::getWarpSize()] number of threads
  // a block reduction is performed here.
  if (NumThreads > mapping::getWarpSize()) {
    // Gather all the reduced values from each warp
    // to the first warp.
    cpyFct(reduce_data, NumWarps);

    if (BlockThreadId < mapping::getWarpSize())
      gpu_irregular_warp_reduce(reduce_data, shflFct, NumWarps, BlockThreadId);
  }

  // In Generic and in SPMD mode block thread Id 0 is what we want.
  // It's either the main thread in SPMD mode or the "acting" main thread in the
  // parallel region.
  return BlockThreadId == 0;
}

uint32_t roundToWarpsize(uint32_t s) {
  if (s < mapping::getWarpSize())
    return 1;
  return (s & ~(unsigned)(mapping::getWarpSize() - 1));
}

uint32_t kmpcMin(uint32_t x, uint32_t y) { return x < y ? x : y; }

} // namespace

extern "C" {
int32_t __kmpc_nvptx_parallel_reduce_nowait_v2(
    IdentTy *Loc, int32_t TId, int32_t num_vars, uint64_t reduce_size,
    void *reduce_data, ShuffleReductFnTy shflFct, InterWarpCopyFnTy cpyFct) {
  return nvptx_parallel_reduce_nowait(reduce_data, shflFct, cpyFct);
}

/// Mostly like _v2 but with the builtin assumption that we have less than
/// num_of_records (by default 1024) teams.
int32_t __kmpc_nvptx_teams_reduce_nowait_v3(
    IdentTy *Loc, int32_t TId, void *__restrict__ GlobalBuffer,
    uint32_t num_of_records, void *reduce_data, ShuffleReductFnTy shflFct,
    InterWarpCopyFnTy cpyFct, ListGlobalFnTy lgcpyFct, ListGlobalFnTy lgredFct,
    ListGlobalFnTy glcpyFct, ListGlobalFnTy glredFct) {
  // Terminate all threads in non-SPMD mode except for the main thread.
  uint32_t ThreadId = mapping::getThreadIdInBlock();
  if (mapping::isGenericMode()) {
    if (!mapping::isMainThreadInGenericMode())
      return 0;
    ThreadId = 0;
  }

  uint32_t &Cnt = state::getKernelLaunchEnvironment().ReductionCnt;

  // In non-generic mode all workers participate in the teams reduction.
  // In generic mode only the team main participates in the teams
  // reduction because the workers are waiting for parallel work.
  uint32_t NumThreads = omp_get_num_threads();
  uint32_t TeamId = omp_get_team_num();
  uint32_t NumTeams = omp_get_num_teams();
  static unsigned SHARED(ChunkTeamCount);

  // Block progress for teams greater than the current upper
  // limit. We always only allow a number of teams less or equal
  // to the number of slots in the buffer.
  bool IsMain = (ThreadId == 0);

  if (IsMain) {
    lgcpyFct(GlobalBuffer, TeamId, reduce_data);

    // Propagate the memory writes above to the world.
    fence::kernel(atomic::release);

    // Increment team counter.
    // This counter is incremented by all teams in the current
    // BUFFER_SIZE chunk.
    ChunkTeamCount = atomic::inc(&Cnt, NumTeams, atomic::acq_rel,
                                 atomic::MemScopeTy::device);
  }

  // Synchronize in SPMD mode as in generic mode all but 1 threads are in the
  // state machine.
  if (mapping::isSPMDMode())
    synchronize::threadsAligned(atomic::acq_rel);

  // Each thread will have a local struct containing the values to be
  // reduced:
  //      1. do reduction within each warp.
  //      2. do reduction across warps.
  //      3. write the final result to the main reduction variable
  //         by returning 1 in the thread holding the reduction result.

  // Check if this is the very last team.
  if (ChunkTeamCount != NumTeams - 1)
    return 0;

  // Last team processing.
  NumThreads = roundToWarpsize(kmpcMin(NumThreads, NumTeams));
  if (ThreadId >= NumThreads)
    return 0;

  // Ensure we see the global memory writes by other teams
  fence::kernel(atomic::aquire);

  // Load from buffer and reduce.
  glcpyFct(GlobalBuffer, ThreadId, reduce_data);
  for (uint32_t i = NumThreads + ThreadId; i < NumTeams; i += NumThreads)
    glredFct(GlobalBuffer, i, reduce_data);

  // Reduce across warps to the warp main.
  gpu_regular_warp_reduce(reduce_data, shflFct);

  uint32_t ActiveThreads = kmpcMin(NumTeams, NumThreads);
  uint32_t WarpsNeeded =
      (ActiveThreads + mapping::getWarpSize() - 1) / mapping::getWarpSize();
  // Gather all the reduced values from each warp
  // to the first warp.
  cpyFct(reduce_data, WarpsNeeded);

  if (mapping::getWarpIdInBlock() == 0)
    gpu_irregular_warp_reduce(reduce_data, shflFct, WarpsNeeded, ThreadId);

  return IsMain;
}

int32_t __kmpc_nvptx_teams_reduce_nowait_v2(
    IdentTy *Loc, int32_t TId, void *GlobalBuffer, uint32_t num_of_records,
    void *reduce_data, ShuffleReductFnTy shflFct, InterWarpCopyFnTy cpyFct,
    ListGlobalFnTy lgcpyFct, ListGlobalFnTy lgredFct, ListGlobalFnTy glcpyFct,
    ListGlobalFnTy glredFct) {
  // The first check is a compile time constant, the second one a runtime check.
  // If the first one succeeds we will use the specialized version.
  if ((state::getKernelEnvironment().Configuration.MaxTeams >= 0 &&
       state::getKernelEnvironment().Configuration.MaxTeams <= num_of_records &&
       num_of_records == 1024) ||
      (omp_get_num_teams() <= num_of_records))
    return __kmpc_nvptx_teams_reduce_nowait_v3(
        Loc, TId, GlobalBuffer, num_of_records, reduce_data, shflFct, cpyFct,
        lgcpyFct, lgredFct, glcpyFct, glredFct);

  // Terminate all threads in non-SPMD mode except for the master thread.
  uint32_t ThreadId = mapping::getThreadIdInBlock();
  if (mapping::isGenericMode()) {
    if (!mapping::isMainThreadInGenericMode())
      return 0;
    ThreadId = 0;
  }

  uint32_t &IterCnt = state::getKernelLaunchEnvironment().ReductionIterCnt;
  uint32_t &Cnt = state::getKernelLaunchEnvironment().ReductionCnt;

  // In non-generic mode all workers participate in the teams reduction.
  // In generic mode only the team master participates in the teams
  // reduction because the workers are waiting for parallel work.
  uint32_t NumThreads = omp_get_num_threads();
  uint32_t TeamId = omp_get_team_num();
  uint32_t NumTeams = omp_get_num_teams();
  static unsigned SHARED(Bound);
  static unsigned SHARED(ChunkTeamCount);

  // Block progress for teams greater than the current upper
  // limit. We always only allow a number of teams less or equal
  // to the number of slots in the buffer.
  bool IsMaster = (ThreadId == 0);
  while (IsMaster) {
    Bound = atomic::load(&IterCnt, atomic::aquire);
    if (TeamId < Bound + num_of_records)
      break;
  }

  if (IsMaster) {
    int ModBockId = TeamId % num_of_records;
    if (TeamId < num_of_records) {
      lgcpyFct(GlobalBuffer, ModBockId, reduce_data);
    } else
      lgredFct(GlobalBuffer, ModBockId, reduce_data);

    // Propagate the memory writes above to the world.
    fence::kernel(atomic::release);

    // Increment team counter.
    // This counter is incremented by all teams in the current
    // num_of_records chunk.
    ChunkTeamCount = atomic::inc(&Cnt, num_of_records - 1u, atomic::seq_cst,
                                 atomic::MemScopeTy::device);
  }

  // Synchronize in SPMD mode as in generic mode all but 1 threads are in the
  // state machine.
  if (mapping::isSPMDMode())
    synchronize::threadsAligned(atomic::acq_rel);

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
  unsigned NumRecs = kmpcMin(NumTeams, uint32_t(num_of_records));
  if (ChunkTeamCount == NumTeams - Bound - 1) {
    // Ensure we see the global memory writes by other teams
    fence::kernel(atomic::aquire);

    //
    // Last team processing.
    //
    if (ThreadId >= NumRecs)
      return 0;
    NumThreads = roundToWarpsize(kmpcMin(NumThreads, NumRecs));
    if (ThreadId >= NumThreads)
      return 0;

    // Load from buffer and reduce.
    glcpyFct(GlobalBuffer, ThreadId, reduce_data);
    for (uint32_t i = NumThreads + ThreadId; i < NumRecs; i += NumThreads)
      glredFct(GlobalBuffer, i, reduce_data);

    // Reduce across warps to the warp master.
    if (NumThreads > 1) {
      gpu_regular_warp_reduce(reduce_data, shflFct);

      // When we have more than [mapping::getWarpSize()] number of threads
      // a block reduction is performed here.
      uint32_t ActiveThreads = kmpcMin(NumRecs, NumThreads);
      if (ActiveThreads > mapping::getWarpSize()) {
        uint32_t WarpsNeeded = (ActiveThreads + mapping::getWarpSize() - 1) /
                               mapping::getWarpSize();
        // Gather all the reduced values from each warp
        // to the first warp.
        cpyFct(reduce_data, WarpsNeeded);

        uint32_t WarpId = ThreadId / mapping::getWarpSize();
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
    atomic::add(&IterCnt, uint32_t(num_of_records), atomic::seq_cst);
  }

  return 0;
}
}

void *__kmpc_reduction_get_fixed_buffer() {
  return state::getKernelLaunchEnvironment().ReductionBuffer;
}

#pragma omp end declare target
