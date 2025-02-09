//===------- Mapping.cpp - OpenMP device runtime mapping helpers -- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "Mapping.h"
#include "DeviceTypes.h"
#include "DeviceUtils.h"
#include "Interface.h"
#include "State.h"

#include <gpuintrin.h>

#include "llvm/Frontend/OpenMP/OMPGridValues.h"

using namespace ompx;

///{

static bool isInLastWarp() {
  uint32_t MainTId = (mapping::getNumberOfThreadsInBlock() - 1) &
                     ~(mapping::getWarpSize() - 1);
  return mapping::getThreadIdInBlock() == MainTId;
}

bool mapping::isMainThreadInGenericMode(bool IsSPMD) {
  if (IsSPMD || icv::Level)
    return false;

  // Check if this is the last warp in the block.
  return isInLastWarp();
}

bool mapping::isMainThreadInGenericMode() {
  return mapping::isMainThreadInGenericMode(mapping::isSPMDMode());
}

bool mapping::isInitialThreadInLevel0(bool IsSPMD) {
  if (IsSPMD)
    return mapping::getThreadIdInBlock() == 0;
  return isInLastWarp();
}

bool mapping::isLeaderInWarp() {
  return __gpu_is_first_in_lane(__gpu_lane_mask());
}

LaneMaskTy mapping::activemask() { return __gpu_lane_mask(); }

LaneMaskTy mapping::lanemaskLT() {
#ifdef __NVPTX__
  return __nvvm_read_ptx_sreg_lanemask_lt();
#else
  uint32_t Lane = __gpu_lane_id();
  int64_t Ballot = __gpu_lane_mask();
  uint64_t Mask = ((uint64_t)1 << Lane) - (uint64_t)1;
  return Mask & Ballot;
#endif
}

LaneMaskTy mapping::lanemaskGT() {
#ifdef __NVPTX__
  return __nvvm_read_ptx_sreg_lanemask_gt();
#else
  uint32_t Lane = __gpu_lane_id();
  if (Lane == (__gpu_num_lanes() - 1))
    return 0;
  int64_t Ballot = __gpu_lane_mask();
  uint64_t Mask = (~((uint64_t)0)) << (Lane + 1);
  return Mask & Ballot;
#endif
}

uint32_t mapping::getThreadIdInWarp() {
  uint32_t ThreadIdInWarp = __gpu_lane_id();
  return ThreadIdInWarp;
}

uint32_t mapping::getThreadIdInBlock(int32_t Dim) {
  uint32_t ThreadIdInBlock = __gpu_thread_id(Dim);
  return ThreadIdInBlock;
}

uint32_t mapping::getWarpSize() { return __gpu_num_lanes(); }

uint32_t mapping::getMaxTeamThreads(bool IsSPMD) {
  uint32_t BlockSize = mapping::getNumberOfThreadsInBlock();
  // If we are in SPMD mode, remove one warp.
  return BlockSize - (!IsSPMD * __gpu_num_lanes());
}
uint32_t mapping::getMaxTeamThreads() {
  return mapping::getMaxTeamThreads(mapping::isSPMDMode());
}

uint32_t mapping::getNumberOfThreadsInBlock(int32_t Dim) {
  return __gpu_num_threads(Dim);
}

uint32_t mapping::getNumberOfThreadsInKernel() {
  return __gpu_num_threads(mapping::DIM_X) * __gpu_num_threads(mapping::DIM_Y) *
         __gpu_num_threads(mapping::DIM_Z);
}

uint32_t mapping::getWarpIdInBlock() {
  return __gpu_thread_id(mapping::DIM_X) / __gpu_num_lanes();
}

uint32_t mapping::getBlockIdInKernel(int32_t Dim) {
  return __gpu_block_id(Dim);
}

uint32_t mapping::getNumberOfWarpsInBlock() {
  return __gpu_block_id(mapping::DIM_X) / __gpu_num_lanes();
}

uint32_t mapping::getNumberOfBlocksInKernel(int32_t Dim) {
  return __gpu_num_blocks(Dim);
}

uint32_t mapping::getNumberOfProcessorElements() {
  return static_cast<uint32_t>(config::getHardwareParallelism());
}

///}

/// Execution mode
///
///{

// TODO: This is a workaround for initialization coming from kernels outside of
//       the TU. We will need to solve this more correctly in the future.
[[clang::loader_uninitialized, gnu::weak]] int __gpu_local IsSPMDMode;

void mapping::init(bool IsSPMD) {
  if (mapping::isInitialThreadInLevel0(IsSPMD))
    IsSPMDMode = IsSPMD;
}

bool mapping::isSPMDMode() { return IsSPMDMode; }

bool mapping::isGenericMode() { return !isSPMDMode(); }
///}

extern "C" {
[[gnu::noinline]] uint32_t __kmpc_get_hardware_thread_id_in_block() {
  return mapping::getThreadIdInBlock();
}

[[gnu::noinline]] uint32_t __kmpc_get_hardware_num_threads_in_block() {
  return mapping::getNumberOfThreadsInBlock(mapping::DIM_X);
}

[[gnu::noinline]] uint32_t __kmpc_get_warp_size() { return __gpu_num_lanes(); }
}

#define _TGT_KERNEL_LANGUAGE(NAME, MAPPER_NAME)                                \
  extern "C" int ompx_##NAME(int Dim) { return mapping::MAPPER_NAME(Dim); }

_TGT_KERNEL_LANGUAGE(thread_id, getThreadIdInBlock)
_TGT_KERNEL_LANGUAGE(block_id, getBlockIdInKernel)
_TGT_KERNEL_LANGUAGE(block_dim, getNumberOfThreadsInBlock)
_TGT_KERNEL_LANGUAGE(grid_dim, getNumberOfBlocksInKernel)

extern "C" {
uint64_t ompx_ballot_sync(uint64_t mask, int pred) {
  return utils::ballotSync(mask, pred);
}

int ompx_shfl_down_sync_i(uint64_t mask, int var, unsigned delta, int width) {
  return utils::shuffleDown(mask, var, delta, width);
}

float ompx_shfl_down_sync_f(uint64_t mask, float var, unsigned delta,
                            int width) {
  return utils::bitCast<float>(
      utils::shuffleDown(mask, utils::bitCast<int32_t>(var), delta, width));
}

long ompx_shfl_down_sync_l(uint64_t mask, long var, unsigned delta, int width) {
  return utils::shuffleDown(mask, utils::bitCast<int64_t>(var), delta, width);
}

double ompx_shfl_down_sync_d(uint64_t mask, double var, unsigned delta,
                             int width) {
  return utils::bitCast<double>(
      utils::shuffleDown(mask, utils::bitCast<int64_t>(var), delta, width));
}
}
