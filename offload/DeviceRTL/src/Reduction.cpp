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
#include "DeviceTypes.h"
#include "DeviceUtils.h"
#include "Interface.h"
#include "Mapping.h"
#include "State.h"
#include "Synchronization.h"

using namespace ompx;

namespace {

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

static uint32_t gpu_irregular_simd_reduce(void *reduce_data,
                                          ShuffleReductFnTy shflFct) {
  uint32_t size, remote_id, physical_lane_id;
  physical_lane_id = mapping::getThreadIdInBlock() % mapping::getWarpSize();
  __kmpc_impl_lanemask_t lanemask_lt = mapping::lanemaskLT();
  __kmpc_impl_lanemask_t Liveness = mapping::activemask();
  uint32_t logical_lane_id = utils::popc(Liveness & lanemask_lt) * 2;
  __kmpc_impl_lanemask_t lanemask_gt = mapping::lanemaskGT();
  do {
    Liveness = mapping::activemask();
    remote_id = utils::ffs(Liveness & lanemask_gt);
    size = utils::popc(Liveness);
    logical_lane_id /= 2;
    shflFct(reduce_data, /*LaneId =*/logical_lane_id,
            /*Offset=*/remote_id - 1 - physical_lane_id, /*AlgoVersion=*/2);
  } while (logical_lane_id % 2 == 0 && size > 1);
  return (logical_lane_id == 0);
}

static int32_t nvptx_parallel_reduce_nowait(void *reduce_data,
                                            ShuffleReductFnTy shflFct,
                                            InterWarpCopyFnTy cpyFct) {
  uint32_t BlockThreadId = mapping::getThreadIdInBlock();
  if (mapping::isMainThreadInGenericMode(/*IsSPMD=*/false))
    BlockThreadId = 0;
  uint32_t NumThreads = omp_get_num_threads();
  if (NumThreads == 1)
    return 1;

    //
    // This reduce function handles reduction within a team. It handles
    // parallel regions in both L1 and L2 parallelism levels. It also
    // supports Generic, SPMD, and NoOMP modes.
    //
    // 1. Reduce within a warp.
    // 2. Warp master copies value to warp 0 via shared memory.
    // 3. Warp 0 reduces to a single value.
    // 4. The reduced value is available in the thread that returns 1.
    //

#if __has_builtin(__nvvm_reflect)
  if (__nvvm_reflect("__CUDA_ARCH") >= 700) {
    uint32_t WarpsNeeded =
        (NumThreads + mapping::getWarpSize() - 1) / mapping::getWarpSize();
    uint32_t WarpId = mapping::getWarpIdInBlock();

    // Volta execution model:
    // For the Generic execution mode a parallel region either has 1 thread and
    // beyond that, always a multiple of 32. For the SPMD execution mode we may
    // have any number of threads.
    if ((NumThreads % mapping::getWarpSize() == 0) ||
        (WarpId < WarpsNeeded - 1))
      gpu_regular_warp_reduce(reduce_data, shflFct);
    else if (NumThreads > 1) // Only SPMD execution mode comes thru this case.
      gpu_irregular_warp_reduce(
          reduce_data, shflFct,
          /*LaneCount=*/NumThreads % mapping::getWarpSize(),
          /*LaneId=*/mapping::getThreadIdInBlock() % mapping::getWarpSize());

    // When we have more than [mapping::getWarpSize()] number of threads
    // a block reduction is performed here.
    //
    // Only L1 parallel region can enter this if condition.
    if (NumThreads > mapping::getWarpSize()) {
      // Gather all the reduced values from each warp
      // to the first warp.
      cpyFct(reduce_data, WarpsNeeded);

      if (WarpId == 0)
        gpu_irregular_warp_reduce(reduce_data, shflFct, WarpsNeeded,
                                  BlockThreadId);
    }
    return BlockThreadId == 0;
  }
#endif
  __kmpc_impl_lanemask_t Liveness = mapping::activemask();
  if (Liveness == lanes::All) // Full warp
    gpu_regular_warp_reduce(reduce_data, shflFct);
  else if (!(Liveness & (Liveness + 1))) // Partial warp but contiguous lanes
    gpu_irregular_warp_reduce(reduce_data, shflFct,
                              /*LaneCount=*/utils::popc(Liveness),
                              /*LaneId=*/mapping::getThreadIdInBlock() %
                                  mapping::getWarpSize());
  else { // Dispersed lanes. Only threads in L2
         // parallel region may enter here; return
         // early.
    return gpu_irregular_simd_reduce(reduce_data, shflFct);
  }

  // When we have more than [mapping::getWarpSize()] number of threads
  // a block reduction is performed here.
  //
  // Only L1 parallel region can enter this if condition.
  if (NumThreads > mapping::getWarpSize()) {
    uint32_t WarpsNeeded =
        (NumThreads + mapping::getWarpSize() - 1) / mapping::getWarpSize();
    // Gather all the reduced values from each warp
    // to the first warp.
    cpyFct(reduce_data, WarpsNeeded);

    uint32_t WarpId = BlockThreadId / mapping::getWarpSize();
    if (WarpId == 0)
      gpu_irregular_warp_reduce(reduce_data, shflFct, WarpsNeeded,
                                BlockThreadId);

    return BlockThreadId == 0;
  }

  // Get the OMP thread Id. This is different from BlockThreadId in the case
  // of an L2 parallel region.
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
int32_t __kmpc_nvptx_parallel_reduce_nowait_v2(IdentTy *Loc,
                                               uint64_t reduce_data_size,
                                               void *reduce_data,
                                               ShuffleReductFnTy shflFct,
                                               InterWarpCopyFnTy cpyFct) {
  return nvptx_parallel_reduce_nowait(reduce_data, shflFct, cpyFct);
}

int32_t __kmpc_nvptx_teams_reduce_nowait_v2(
    IdentTy *Loc, void *GlobalBuffer, uint32_t num_of_records,
    uint64_t reduce_data_size, void *reduce_data, ShuffleReductFnTy shflFct,
    InterWarpCopyFnTy cpyFct, ListGlobalFnTy lgcpyFct, ListGlobalFnTy lgredFct,
    ListGlobalFnTy glcpyFct, ListGlobalFnTy glredFct) {
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
    Bound = atomic::load(&IterCnt, atomic::acquire);
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
    fence::kernel(atomic::acquire);

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
