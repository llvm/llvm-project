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
#include "Interface.h"
#include "State.h"
#include "Types.h"
#include "Utils.h"

#pragma omp begin declare target device_type(nohost)

#include "llvm/Frontend/OpenMP/OMPGridValues.h"

using namespace ompx;

namespace ompx {
namespace impl {

// Forward declarations defined to be defined for AMDGCN and NVPTX.
const llvm::omp::GV &getGridValue();
LaneMaskTy activemask();
LaneMaskTy lanemaskLT();
LaneMaskTy lanemaskGT();
uint32_t getThreadIdInWarp();
uint32_t getThreadIdInBlock(int32_t Dim);
uint32_t getNumberOfThreadsInBlock(int32_t Dim);
uint32_t getNumberOfThreadsInKernel();
uint32_t getBlockIdInKernel(int32_t Dim);
uint32_t getNumberOfBlocksInKernel(int32_t Dim);
uint32_t getWarpIdInBlock();
uint32_t getNumberOfWarpsInBlock();

/// AMDGCN Implementation
///
///{
#pragma omp begin declare variant match(device = {arch(amdgcn)})

const llvm::omp::GV &getGridValue() {
  return llvm::omp::getAMDGPUGridValues<__AMDGCN_WAVEFRONT_SIZE>();
}

uint32_t getNumberOfThreadsInBlock(int32_t Dim) {
  switch (Dim) {
  case 0:
    return __builtin_amdgcn_workgroup_size_x();
  case 1:
    return __builtin_amdgcn_workgroup_size_y();
  case 2:
    return __builtin_amdgcn_workgroup_size_z();
  };
  UNREACHABLE("Dim outside range!");
}

LaneMaskTy activemask() { return __builtin_amdgcn_read_exec(); }

LaneMaskTy lanemaskLT() {
  uint32_t Lane = mapping::getThreadIdInWarp();
  int64_t Ballot = mapping::activemask();
  uint64_t Mask = ((uint64_t)1 << Lane) - (uint64_t)1;
  return Mask & Ballot;
}

LaneMaskTy lanemaskGT() {
  uint32_t Lane = mapping::getThreadIdInWarp();
  if (Lane == (mapping::getWarpSize() - 1))
    return 0;
  int64_t Ballot = mapping::activemask();
  uint64_t Mask = (~((uint64_t)0)) << (Lane + 1);
  return Mask & Ballot;
}

uint32_t getThreadIdInWarp() {
  return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}

uint32_t getThreadIdInBlock(int32_t Dim) {
  switch (Dim) {
  case 0:
    return __builtin_amdgcn_workitem_id_x();
  case 1:
    return __builtin_amdgcn_workitem_id_y();
  case 2:
    return __builtin_amdgcn_workitem_id_z();
  };
  UNREACHABLE("Dim outside range!");
}

uint32_t getNumberOfThreadsInKernel() {
  return __builtin_amdgcn_grid_size_x() * __builtin_amdgcn_grid_size_y() *
         __builtin_amdgcn_grid_size_z();
}

uint32_t getBlockIdInKernel(int32_t Dim) {
  switch (Dim) {
  case 0:
    return __builtin_amdgcn_workgroup_id_x();
  case 1:
    return __builtin_amdgcn_workgroup_id_y();
  case 2:
    return __builtin_amdgcn_workgroup_id_z();
  };
  UNREACHABLE("Dim outside range!");
}

uint32_t getNumberOfBlocksInKernel(int32_t Dim) {
  switch (Dim) {
  case 0:
    return __builtin_amdgcn_grid_size_x() / __builtin_amdgcn_workgroup_size_x();
  case 1:
    return __builtin_amdgcn_grid_size_y() / __builtin_amdgcn_workgroup_size_y();
  case 2:
    return __builtin_amdgcn_grid_size_z() / __builtin_amdgcn_workgroup_size_z();
  };
  UNREACHABLE("Dim outside range!");
}

uint32_t getWarpIdInBlock() {
  return impl::getThreadIdInBlock(mapping::DIM_X) / mapping::getWarpSize();
}

uint32_t getNumberOfWarpsInBlock() {
  return mapping::getNumberOfThreadsInBlock() / mapping::getWarpSize();
}

#pragma omp end declare variant
///}

/// NVPTX Implementation
///
///{
#pragma omp begin declare variant match(                                       \
        device = {arch(nvptx, nvptx64)},                                       \
            implementation = {extension(match_any)})

uint32_t getNumberOfThreadsInBlock(int32_t Dim) {
  switch (Dim) {
  case 0:
    return __nvvm_read_ptx_sreg_ntid_x();
  case 1:
    return __nvvm_read_ptx_sreg_ntid_y();
  case 2:
    return __nvvm_read_ptx_sreg_ntid_z();
  };
  UNREACHABLE("Dim outside range!");
}

const llvm::omp::GV &getGridValue() { return llvm::omp::NVPTXGridValues; }

LaneMaskTy activemask() {
  unsigned int Mask;
  asm("activemask.b32 %0;" : "=r"(Mask));
  return Mask;
}

LaneMaskTy lanemaskLT() {
  __kmpc_impl_lanemask_t Res;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(Res));
  return Res;
}

LaneMaskTy lanemaskGT() {
  __kmpc_impl_lanemask_t Res;
  asm("mov.u32 %0, %%lanemask_gt;" : "=r"(Res));
  return Res;
}

uint32_t getThreadIdInBlock(int32_t Dim) {
  switch (Dim) {
  case 0:
    return __nvvm_read_ptx_sreg_tid_x();
  case 1:
    return __nvvm_read_ptx_sreg_tid_y();
  case 2:
    return __nvvm_read_ptx_sreg_tid_z();
  };
  UNREACHABLE("Dim outside range!");
}

uint32_t getThreadIdInWarp() {
  return impl::getThreadIdInBlock(mapping::DIM_X) &
         (mapping::getWarpSize() - 1);
}

uint32_t getBlockIdInKernel(int32_t Dim) {
  switch (Dim) {
  case 0:
    return __nvvm_read_ptx_sreg_ctaid_x();
  case 1:
    return __nvvm_read_ptx_sreg_ctaid_y();
  case 2:
    return __nvvm_read_ptx_sreg_ctaid_z();
  };
  UNREACHABLE("Dim outside range!");
}

uint32_t getNumberOfBlocksInKernel(int32_t Dim) {
  switch (Dim) {
  case 0:
    return __nvvm_read_ptx_sreg_nctaid_x();
  case 1:
    return __nvvm_read_ptx_sreg_nctaid_y();
  case 2:
    return __nvvm_read_ptx_sreg_nctaid_z();
  };
  UNREACHABLE("Dim outside range!");
}

uint32_t getNumberOfThreadsInKernel() {
  return impl::getNumberOfThreadsInBlock(0) *
         impl::getNumberOfBlocksInKernel(0) *
         impl::getNumberOfThreadsInBlock(1) *
         impl::getNumberOfBlocksInKernel(1) *
         impl::getNumberOfThreadsInBlock(2) *
         impl::getNumberOfBlocksInKernel(2);
}

uint32_t getWarpIdInBlock() {
  return impl::getThreadIdInBlock(mapping::DIM_X) / mapping::getWarpSize();
}

uint32_t getNumberOfWarpsInBlock() {
  return (mapping::getNumberOfThreadsInBlock() + mapping::getWarpSize() - 1) /
         mapping::getWarpSize();
}

#pragma omp end declare variant
///}

uint32_t getWarpSize() { return getGridValue().GV_Warp_Size; }

} // namespace impl
} // namespace ompx

/// We have to be deliberate about the distinction of `mapping::` and `impl::`
/// below to avoid repeating assumptions or including irrelevant ones.
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
  __kmpc_impl_lanemask_t Active = mapping::activemask();
  __kmpc_impl_lanemask_t LaneMaskLT = mapping::lanemaskLT();
  return utils::popc(Active & LaneMaskLT) == 0;
}

LaneMaskTy mapping::activemask() { return impl::activemask(); }

LaneMaskTy mapping::lanemaskLT() { return impl::lanemaskLT(); }

LaneMaskTy mapping::lanemaskGT() { return impl::lanemaskGT(); }

uint32_t mapping::getThreadIdInWarp() {
  uint32_t ThreadIdInWarp = impl::getThreadIdInWarp();
  ASSERT(ThreadIdInWarp < impl::getWarpSize(), nullptr);
  return ThreadIdInWarp;
}

uint32_t mapping::getThreadIdInBlock(int32_t Dim) {
  uint32_t ThreadIdInBlock = impl::getThreadIdInBlock(Dim);
  return ThreadIdInBlock;
}

uint32_t mapping::getWarpSize() { return impl::getWarpSize(); }

uint32_t mapping::getMaxTeamThreads(bool IsSPMD) {
  uint32_t BlockSize = mapping::getNumberOfThreadsInBlock();
  // If we are in SPMD mode, remove one warp.
  return BlockSize - (!IsSPMD * impl::getWarpSize());
}
uint32_t mapping::getMaxTeamThreads() {
  return mapping::getMaxTeamThreads(mapping::isSPMDMode());
}

uint32_t mapping::getNumberOfThreadsInBlock(int32_t Dim) {
  return impl::getNumberOfThreadsInBlock(Dim);
}

uint32_t mapping::getNumberOfThreadsInKernel() {
  return impl::getNumberOfThreadsInKernel();
}

uint32_t mapping::getWarpIdInBlock() {
  uint32_t WarpID = impl::getWarpIdInBlock();
  ASSERT(WarpID < impl::getNumberOfWarpsInBlock(), nullptr);
  return WarpID;
}

uint32_t mapping::getBlockIdInKernel(int32_t Dim) {
  uint32_t BlockId = impl::getBlockIdInKernel(Dim);
  ASSERT(BlockId < impl::getNumberOfBlocksInKernel(Dim), nullptr);
  return BlockId;
}

uint32_t mapping::getNumberOfWarpsInBlock() {
  uint32_t NumberOfWarpsInBlocks = impl::getNumberOfWarpsInBlock();
  ASSERT(impl::getWarpIdInBlock() < NumberOfWarpsInBlocks, nullptr);
  return NumberOfWarpsInBlocks;
}

uint32_t mapping::getNumberOfBlocksInKernel(int32_t Dim) {
  uint32_t NumberOfBlocks = impl::getNumberOfBlocksInKernel(Dim);
  ASSERT(impl::getBlockIdInKernel(Dim) < NumberOfBlocks, nullptr);
  return NumberOfBlocks;
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
int __attribute__((weak)) SHARED(IsSPMDMode);

void mapping::init(bool IsSPMD) {
  if (mapping::isInitialThreadInLevel0(IsSPMD))
    IsSPMDMode = IsSPMD;
}

bool mapping::isSPMDMode() { return IsSPMDMode; }

bool mapping::isGenericMode() { return !isSPMDMode(); }
///}

extern "C" {
__attribute__((noinline)) uint32_t __kmpc_get_hardware_thread_id_in_block() {
  return mapping::getThreadIdInBlock();
}

__attribute__((noinline)) uint32_t __kmpc_get_hardware_num_threads_in_block() {
  return impl::getNumberOfThreadsInBlock(mapping::DIM_X);
}

__attribute__((noinline)) uint32_t __kmpc_get_warp_size() {
  return impl::getWarpSize();
}
}

#define _TGT_KERNEL_LANGUAGE(NAME, MAPPER_NAME)                                \
  extern "C" int ompx_##NAME(int Dim) { return mapping::MAPPER_NAME(Dim); }

_TGT_KERNEL_LANGUAGE(thread_id, getThreadIdInBlock)
_TGT_KERNEL_LANGUAGE(block_id, getBlockIdInKernel)
_TGT_KERNEL_LANGUAGE(block_dim, getNumberOfThreadsInBlock)
_TGT_KERNEL_LANGUAGE(grid_dim, getNumberOfBlocksInKernel)

#pragma omp end declare target
