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

#include "DeviceTypes.h"
#include "Interface.h"
#include "Mapping.h"
#include "State.h"
#include "Synchronization.h"

using namespace ompx;

namespace {
static constexpr uint32_t kmpc_min(uint32_t a, uint32_t b) {
  return a < b ? a : b;
}

[[clang::always_inline]]
void gpu_regular_warp_reduce(void *reduce_data, ShuffleReductFnTy shflFct) {
  for (uint32_t mask = mapping::getWarpSize() / 2; mask > 0; mask /= 2) {
    shflFct(reduce_data, /*LaneId - not used= */ 0,
            /*Offset = */ mask, /*AlgoVersion=*/0);
  }
}

[[clang::always_inline]]
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

[[clang::always_inline]]
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
    remote_id = utils::ctz(Liveness & lanemask_gt);
    size = utils::popc(Liveness);
    logical_lane_id /= 2;
    shflFct(reduce_data, /*LaneId =*/logical_lane_id,
            /*Offset=*/remote_id - physical_lane_id, /*AlgoVersion=*/2);
  } while (logical_lane_id % 2 == 0 && size > 1);
  return (logical_lane_id == 0);
}

// Reduction within a block on the GPU.
//
// Template parameters:
// - checkLiveness: Whether to check the liveness of the lanes. This is only
//                  useful if gpu_block_reduce is called in a context where
//                  L2 parallel regions are possible.
// Parameters:
// - reduce_data: Pointer to the reduction data
// - shflFct:     Shuffle reduction function
// - cpyFct:      Inter-warp copy function (copies data from each warp's thread
//                0 to the lanes of the zeroth warp)
// - NumValues:   Number of values to reduce / threads to consider
// - ThreadId:    Thread ID in block (getThreadIdInBlock() in SPMD and 0 in
//                Generic mode)
//
// Returns:
// - 1 if the thread is the zeroth thread of the block
// - 0 otherwise
template <bool checkLiveness = true>
[[clang::always_inline]]
static uint32_t gpu_block_reduce(void *reduce_data, ShuffleReductFnTy shflFct,
                                 InterWarpCopyFnTy cpyFct, uint32_t NumValues,
                                 uint32_t BlockThreadId) {
  if (NumValues <= 1)
    return BlockThreadId == 0;

  uint32_t WarpId = BlockThreadId / mapping::getWarpSize();
  uint32_t WarpOffset = WarpId * mapping::getWarpSize();
  // Calculate how many values this warp has to deal with. Cap WarpId *
  // mapping::getWarpSize() at NumValues to avoid underflow.
  uint32_t ActiveLanes =
      WarpOffset < NumValues
          ? kmpc_min(NumValues - WarpOffset, mapping::getWarpSize())
          : 0;

  if constexpr (checkLiveness) {
    __kmpc_impl_lanemask_t Liveness = mapping::activemask();
    // Check for partial warp with non-contiguous lanes.
    if (Liveness != lanes::All && (Liveness & (Liveness + 1))) {
      // Only threads in L2 parallel region may enter here.
      return gpu_irregular_simd_reduce(reduce_data, shflFct);
    }
    ActiveLanes = kmpc_min(ActiveLanes, utils::popc(Liveness));
  }

  if (ActiveLanes < mapping::getWarpSize())
    gpu_irregular_warp_reduce(reduce_data, shflFct, ActiveLanes,
                              BlockThreadId % mapping::getWarpSize());
  else
    gpu_regular_warp_reduce(reduce_data, shflFct);

  // When we have more than [mapping::getWarpSize()] number of threads
  // a block reduction is performed here.
  //
  // Only L1 parallel region can enter this if condition.

  if (NumValues > mapping::getWarpSize()) {
    uint32_t WarpsNeeded =
        (NumValues + mapping::getWarpSize() - 1) / mapping::getWarpSize();
    // Gather all the reduced values from each warp
    // to the first warp.
    cpyFct(reduce_data, WarpsNeeded);

    if (WarpId == 0)
      gpu_irregular_warp_reduce(reduce_data, shflFct, WarpsNeeded,
                                BlockThreadId);
  }

  return BlockThreadId == 0;
}

[[clang::always_inline]]
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

  return gpu_block_reduce(reduce_data, shflFct, cpyFct, NumThreads,
                          BlockThreadId);
}

} // namespace

extern "C" {
[[clang::always_inline]]
int32_t __kmpc_nvptx_parallel_reduce_nowait_v2(IdentTy *Loc,
                                               uint64_t reduce_data_size,
                                               void *reduce_data,
                                               ShuffleReductFnTy shflFct,
                                               InterWarpCopyFnTy cpyFct) {
  return nvptx_parallel_reduce_nowait(reduce_data, shflFct, cpyFct);
}

// Reduction across teams on the GPU.
//
// Parameters:
// - Loc: Location of the reduction
// - reduce_data: Pointer to the reduction data
// - shflFct:  Shuffle reduction function
// - cpyFct:   Inter-warp copy function (copies data from each warp's thread 0
//             to the lanes of the zeroth warp)
// - lgcpyFct: List-global copy function (copies the reduction data from the
//             local thread to the global buffer)
// - glcpyFct: Global copy function (copies the reduction data from the global
//             buffer to the local thread)
// - glredFct: Global reduce function (reduces the reduction data from the
//             global buffer to the local thread)
//
// Returns:
// - 1 if this thread must write the final reduced value back to the shared
//   reduction variable (i.e. thread 0 of the single team when NumTeams == 1,
//   or thread 0 of the last team to finish its partial reduction otherwise).
// - 0 otherwise.
//
[[clang::always_inline]]
int32_t __kmpc_gpu_xteam_reduce_nowait(IdentTy *Loc, void *reduce_data,
                                       ShuffleReductFnTy shflFct,
                                       InterWarpCopyFnTy cpyFct,
                                       ListGlobalFnTy lgcpyFct,
                                       ListGlobalFnTy glcpyFct,
                                       ListGlobalFnTy glredFct) {
  // Terminate all threads in non-SPMD mode except for the master thread.
  uint32_t ThreadId = mapping::getThreadIdInBlock();
  if (mapping::isGenericMode()) {
    if (!mapping::isMainThreadInGenericMode())
      return 0;
    ThreadId = 0;
  }

  // In non-generic mode all workers participate in the teams reduction.
  // In generic mode only the team master participates in the teams
  // reduction because the workers are waiting for parallel work.
  uint32_t NumThreads = omp_get_num_threads();
  uint32_t TeamId = omp_get_team_num();
  uint32_t NumTeams = omp_get_num_teams();

  // Fast path for single-team kernels: no cross-team work required,
  // the team-local reduction already produced the final result.
  if (NumTeams <= 1)
    return ThreadId == 0;

  uint32_t &TeamsDone = state::getKernelLaunchEnvironment().ReductionTeamsDone;
  void *GlobalBuffer = state::getKernelLaunchEnvironment().ReductionBuffer;
  [[clang::loader_uninitialized]] static Local<uint32_t> TeamsDoneResult;

  // Save the team's reduced value in the global buffer and atomically
  // increment the teams-done counter.
  if (ThreadId == 0) {
    lgcpyFct(GlobalBuffer, TeamId, reduce_data);
    TeamsDoneResult = atomic::inc(&TeamsDone, NumTeams - 1u, atomic::acq_rel,
                                  atomic::MemScopeTy::device);
  }

  // This sync is needed so that all threads from last team see the shared teams
  // done counter value and know that they are in the last team.
  if (mapping::isSPMDMode())
    synchronize::threadsAligned(atomic::acq_rel);

  // If teams done counter reaches NumTeams-1, this is the last team.
  if (TeamsDoneResult != NumTeams - 1u)
    return 0;

  // The last team performs final reduction across all team values.
  uint32_t ValidValues = NumThreads < NumTeams ? NumThreads : NumTeams;
  if (ThreadId < ValidValues) {
    // Make sure that global buffer is fresh.
    fence::kernel(atomic::acquire);
    // Get the team values from the global buffer.
    glcpyFct(GlobalBuffer, ThreadId, reduce_data);
    // In case we have more teams than threads, we need to iterate over the
    // remaining teams.
    for (uint32_t I = NumThreads + ThreadId; I < NumTeams; I += NumThreads)
      glredFct(GlobalBuffer, I, reduce_data);
  }

  return gpu_block_reduce<false>(reduce_data, shflFct, cpyFct, ValidValues,
                                 ThreadId);
}
} // extern "C"

void *__kmpc_reduction_get_fixed_buffer() {
  return state::getKernelLaunchEnvironment().ReductionBuffer;
}
