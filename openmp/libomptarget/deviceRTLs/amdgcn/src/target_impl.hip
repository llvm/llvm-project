//===------- target_impl.hip - AMDGCN OpenMP GPU implementation --- HIP -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions of target specific functions
//
//===----------------------------------------------------------------------===//

#include "target_impl.h"

// Implementations initially derived from hcc

static DEVICE uint32_t getLaneId(void) {
  return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}

// Initialized with a 64-bit mask with bits set in positions less than the
// thread's lane number in the warp
DEVICE __kmpc_impl_lanemask_t __kmpc_impl_lanemask_lt() {
  uint32_t lane = getLaneId();
  int64_t ballot = __kmpc_impl_activemask();
  uint64_t mask = ((uint64_t)1 << lane) - (uint64_t)1;
  return mask & ballot;
}

// Initialized with a 64-bit mask with bits set in positions greater than the
// thread's lane number in the warp
DEVICE __kmpc_impl_lanemask_t __kmpc_impl_lanemask_gt() {
  uint32_t lane = getLaneId();
  if (lane == (WARPSIZE - 1))
    return 0;
  uint64_t ballot = __kmpc_impl_activemask();
  uint64_t mask = (~((uint64_t)0)) << (lane + 1);
  return mask & ballot;
}

DEVICE double __kmpc_impl_get_wtick() { return ((double)1E-9); }

EXTERN uint64_t __clock64();
DEVICE double __kmpc_impl_get_wtime() {
  return ((double)1.0 / 745000000.0) * __clock64();
}

// Warp vote function
DEVICE __kmpc_impl_lanemask_t __kmpc_impl_activemask() {
  // 33 is ICMP_NE from llvm/include/llvm/IR/InstrTypes.h
  return __builtin_amdgcn_uicmp(1, 0, 33);
}

DEVICE int32_t __kmpc_impl_shfl_sync(__kmpc_impl_lanemask_t, int32_t var,
                                     int32_t srcLane) {
  int width = WARPSIZE;
  int self = getLaneId();
  int index = srcLane + (self & ~(width - 1));
  return __builtin_amdgcn_ds_bpermute(index << 2, var);
}

DEVICE int32_t __kmpc_impl_shfl_down_sync(__kmpc_impl_lanemask_t, int32_t var,
                                          uint32_t laneDelta, int32_t width) {
  int self = getLaneId();
  int index = self + laneDelta;
  index = (int)(laneDelta + (self & (width - 1))) >= width ? self : index;
  return __builtin_amdgcn_ds_bpermute(index << 2, var);
}

EXTERN uint64_t __ockl_get_local_size(uint32_t);
EXTERN uint64_t __ockl_get_num_groups(uint32_t);
DEVICE int GetNumberOfBlocksInKernel() { return __ockl_get_num_groups(0); }
DEVICE int GetNumberOfThreadsInBlock() { return __ockl_get_local_size(0); }
