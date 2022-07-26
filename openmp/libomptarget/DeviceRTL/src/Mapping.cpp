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

using namespace _OMP;

namespace _OMP {
namespace impl {

// Forward declarations defined to be defined for AMDGCN and NVPTX.
const llvm::omp::GV &getGridValue();
uint32_t getGridDim(uint32_t n, uint16_t d);
uint32_t getWorkgroupDim(uint32_t group_id, uint32_t grid_size,
                         uint16_t group_size);
uint32_t getNumHardwareThreadsInBlock();
LaneMaskTy activemask();
LaneMaskTy lanemaskLT();
LaneMaskTy lanemaskGT();
uint32_t getThreadIdInWarp();
uint32_t getThreadIdInBlock();
uint32_t getKernelSize();
uint32_t getBlockId();
uint32_t getNumberOfBlocks();
uint32_t getWarpId();
uint32_t getNumberOfWarpsInBlock();

/// AMDGCN Implementation
///
///{
#pragma omp begin declare variant match(device = {arch(amdgcn)})

const llvm::omp::GV &getGridValue() {
  return llvm::omp::getAMDGPUGridValues<__AMDGCN_WAVEFRONT_SIZE>();
}

uint32_t getGridDim(uint32_t n, uint16_t d) {
  uint32_t q = n / d;
  return q + (n > q * d);
}

uint32_t getWorkgroupDim(uint32_t group_id, uint32_t grid_size,
                         uint16_t group_size) {
  uint32_t r = grid_size - group_id * group_size;
  return (r < group_size) ? r : group_size;
}

uint32_t getNumHardwareThreadsInBlock() {
  return getWorkgroupDim(__builtin_amdgcn_workgroup_id_x(),
                         __builtin_amdgcn_grid_size_x(),
                         __builtin_amdgcn_workgroup_size_x());
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

uint32_t getThreadIdInBlock() { return __builtin_amdgcn_workitem_id_x(); }

uint32_t getKernelSize() { return __builtin_amdgcn_grid_size_x(); }

uint32_t getBlockId() { return __builtin_amdgcn_workgroup_id_x(); }

uint32_t getNumberOfBlocks() {
  return getGridDim(__builtin_amdgcn_grid_size_x(),
                    __builtin_amdgcn_workgroup_size_x());
}

uint32_t getWarpId() {
  return impl::getThreadIdInBlock() / mapping::getWarpSize();
}

uint32_t getNumberOfWarpsInBlock() {
  return mapping::getBlockSize() / mapping::getWarpSize();
}

#pragma omp end declare variant
///}

/// NVPTX Implementation
///
///{
#pragma omp begin declare variant match(                                       \
    device = {arch(nvptx, nvptx64)}, implementation = {extension(match_any)})

uint32_t getNumHardwareThreadsInBlock() {
  return __nvvm_read_ptx_sreg_ntid_x();
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

uint32_t getThreadIdInBlock() { return __nvvm_read_ptx_sreg_tid_x(); }

uint32_t getThreadIdInWarp() {
  return impl::getThreadIdInBlock() & (mapping::getWarpSize() - 1);
}

uint32_t getKernelSize() {
  return __nvvm_read_ptx_sreg_nctaid_x() *
         mapping::getNumberOfProcessorElements();
}

uint32_t getBlockId() { return __nvvm_read_ptx_sreg_ctaid_x(); }

uint32_t getNumberOfBlocks() { return __nvvm_read_ptx_sreg_nctaid_x(); }

uint32_t getWarpId() {
  return impl::getThreadIdInBlock() / mapping::getWarpSize();
}

uint32_t getNumberOfWarpsInBlock() {
  return (mapping::getBlockSize() + mapping::getWarpSize() - 1) /
         mapping::getWarpSize();
}

#pragma omp end declare variant
///}

uint32_t getWarpSize() { return getGridValue().GV_Warp_Size; }

} // namespace impl
} // namespace _OMP

/// We have to be deliberate about the distinction of `mapping::` and `impl::`
/// below to avoid repeating assumptions or including irrelevant ones.
///{

static bool isInLastWarp() {
  uint32_t MainTId = (mapping::getNumberOfProcessorElements() - 1) &
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
  ASSERT(ThreadIdInWarp < impl::getWarpSize());
  return ThreadIdInWarp;
}

uint32_t mapping::getThreadIdInBlock() {
  uint32_t ThreadIdInBlock = impl::getThreadIdInBlock();
  ASSERT(ThreadIdInBlock < impl::getNumHardwareThreadsInBlock());
  return ThreadIdInBlock;
}

uint32_t mapping::getWarpSize() { return impl::getWarpSize(); }

uint32_t mapping::getBlockSize(bool IsSPMD) {
  uint32_t BlockSize = mapping::getNumberOfProcessorElements() -
                       (!IsSPMD * impl::getWarpSize());
  return BlockSize;
}
uint32_t mapping::getBlockSize() {
  return mapping::getBlockSize(mapping::isSPMDMode());
}

uint32_t mapping::getKernelSize() { return impl::getKernelSize(); }

uint32_t mapping::getWarpId() {
  uint32_t WarpID = impl::getWarpId();
  ASSERT(WarpID < impl::getNumberOfWarpsInBlock());
  return WarpID;
}

uint32_t mapping::getBlockId() {
  uint32_t BlockId = impl::getBlockId();
  ASSERT(BlockId < impl::getNumberOfBlocks());
  return BlockId;
}

uint32_t mapping::getNumberOfWarpsInBlock() {
  uint32_t NumberOfWarpsInBlocks = impl::getNumberOfWarpsInBlock();
  ASSERT(impl::getWarpId() < NumberOfWarpsInBlocks);
  return NumberOfWarpsInBlocks;
}

uint32_t mapping::getNumberOfBlocks() {
  uint32_t NumberOfBlocks = impl::getNumberOfBlocks();
  ASSERT(impl::getBlockId() < NumberOfBlocks);
  return NumberOfBlocks;
}

uint32_t mapping::getNumberOfProcessorElements() {
  uint32_t NumberOfProcessorElements = impl::getNumHardwareThreadsInBlock();
  ASSERT(impl::getThreadIdInBlock() < NumberOfProcessorElements);
  return NumberOfProcessorElements;
}

///}

/// Execution mode
///
///{

// TODO: This is a workaround for initialization coming from kernels outside of
//       the TU. We will need to solve this more correctly in the future.
int __attribute__((weak)) KEEP_ALIVE SHARED(IsSPMDMode);

void mapping::init(bool IsSPMD) {
  if (mapping::isInitialThreadInLevel0(IsSPMD))
    IsSPMDMode = IsSPMD;
}

bool mapping::isSPMDMode() { return IsSPMDMode; }

bool mapping::isGenericMode() { return !isSPMDMode(); }
///}

extern "C" {
uint32_t __kmpc_get_hardware_thread_id_in_block() {
  FunctionTracingRAII();
  return mapping::getThreadIdInBlock();
}

uint32_t __kmpc_get_hardware_num_threads_in_block() {
  FunctionTracingRAII();
  return impl::getNumHardwareThreadsInBlock();
}

uint32_t __kmpc_get_warp_size() {
  FunctionTracingRAII();
  return impl::getWarpSize();
}

__attribute__((noinline)) uint32_t __kmpc_get_hardware_num_blocks() {
  FunctionTracingRAII();
  return impl::getNumberOfBlocks();
}
}
#pragma omp end declare target
