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
#include "State.h"
#include "Types.h"
#include "Utils.h"

#pragma omp declare target

#include "llvm/Frontend/OpenMP/OMPGridValues.h"

using namespace _OMP;

namespace _OMP {
namespace impl {

/// AMDGCN Implementation
///
///{
#pragma omp begin declare variant match(device = {arch(amdgcn)})

constexpr const llvm::omp::GV &getGridValue() {
  return llvm::omp::AMDGPUGridValues;
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

uint32_t getBlockSize() {
  // TODO: verify this logic for generic mode.
  return getWorkgroupDim(__builtin_amdgcn_workgroup_id_x(),
                         __builtin_amdgcn_grid_size_x(),
                         __builtin_amdgcn_workgroup_size_x());
}

uint32_t getKernelSize() { return __builtin_amdgcn_grid_size_x(); }

uint32_t getBlockId() { return __builtin_amdgcn_workgroup_id_x(); }

uint32_t getNumberOfBlocks() {
  return getGridDim(__builtin_amdgcn_grid_size_x(),
                    __builtin_amdgcn_workgroup_size_x());
}

uint32_t getNumberOfProcessorElements() {
  // TODO
  return mapping::getBlockSize();
}

uint32_t getWarpId() {
  return mapping::getThreadIdInBlock() / mapping::getWarpSize();
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

constexpr const llvm::omp::GV &getGridValue() {
  return llvm::omp::NVPTXGridValues;
}

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

uint32_t getThreadIdInWarp() {
  return mapping::getThreadIdInBlock() & (mapping::getWarpSize() - 1);
}

uint32_t getThreadIdInBlock() { return __nvvm_read_ptx_sreg_tid_x(); }

uint32_t getBlockSize() {
  return __nvvm_read_ptx_sreg_ntid_x() -
         (!mapping::isSPMDMode() * mapping::getWarpSize());
}

uint32_t getKernelSize() { return __nvvm_read_ptx_sreg_nctaid_x(); }

uint32_t getBlockId() { return __nvvm_read_ptx_sreg_ctaid_x(); }

uint32_t getNumberOfBlocks() { return __nvvm_read_ptx_sreg_nctaid_x(); }

uint32_t getNumberOfProcessorElements() {
  return __nvvm_read_ptx_sreg_ntid_x();
}

uint32_t getWarpId() {
  return mapping::getThreadIdInBlock() / mapping::getWarpSize();
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

bool mapping::isMainThreadInGenericMode(bool IsSPMD) {
  if (IsSPMD || icv::Level)
    return false;

  // Check if this is the last warp in the block.
  uint32_t MainTId = (mapping::getNumberOfProcessorElements() - 1) &
                     ~(mapping::getWarpSize() - 1);
  return mapping::getThreadIdInBlock() == MainTId;
}

bool mapping::isMainThreadInGenericMode() {
  return mapping::isMainThreadInGenericMode(mapping::isSPMDMode());
}

bool mapping::isLeaderInWarp() {
  __kmpc_impl_lanemask_t Active = mapping::activemask();
  __kmpc_impl_lanemask_t LaneMaskLT = mapping::lanemaskLT();
  return utils::popc(Active & LaneMaskLT) == 0;
}

LaneMaskTy mapping::activemask() { return impl::activemask(); }

LaneMaskTy mapping::lanemaskLT() { return impl::lanemaskLT(); }

LaneMaskTy mapping::lanemaskGT() { return impl::lanemaskGT(); }

uint32_t mapping::getThreadIdInWarp() { return impl::getThreadIdInWarp(); }

uint32_t mapping::getThreadIdInBlock() { return impl::getThreadIdInBlock(); }

uint32_t mapping::getBlockSize() { return impl::getBlockSize(); }

uint32_t mapping::getKernelSize() { return impl::getKernelSize(); }

uint32_t mapping::getBlockId() { return impl::getBlockId(); }

uint32_t mapping::getNumberOfBlocks() { return impl::getNumberOfBlocks(); }

uint32_t mapping::getNumberOfProcessorElements() {
  return impl::getNumberOfProcessorElements();
}

uint32_t mapping::getWarpId() { return impl::getWarpId(); }

uint32_t mapping::getWarpSize() { return impl::getWarpSize(); }

uint32_t mapping::getNumberOfWarpsInBlock() {
  return impl::getNumberOfWarpsInBlock();
}

/// Execution mode
///
///{
static int SHARED(IsSPMDMode);

void mapping::init(bool IsSPMD) {
  if (!mapping::getThreadIdInBlock())
    IsSPMDMode = IsSPMD;
}

bool mapping::isSPMDMode() { return IsSPMDMode; }

bool mapping::isGenericMode() { return !isSPMDMode(); }
///}

extern "C" {
__attribute__((noinline)) uint32_t __kmpc_get_hardware_thread_id_in_block() {
  return mapping::getThreadIdInBlock();
}
}
#pragma omp end declare target
