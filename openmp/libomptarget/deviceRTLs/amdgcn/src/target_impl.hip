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
#include "amdgcn_interface.h"

// Implementations initially derived from hcc

// Initialized with a 64-bit mask with bits set in positions less than the
// thread's lane number in the warp
DEVICE __kmpc_impl_lanemask_t __kmpc_impl_lanemask_lt() {
  uint32_t lane = GetLaneId();
  int64_t ballot = __kmpc_impl_activemask();
  uint64_t mask = ((uint64_t)1 << lane) - (uint64_t)1;
  return mask & ballot;
}

// Initialized with a 64-bit mask with bits set in positions greater than the
// thread's lane number in the warp
DEVICE __kmpc_impl_lanemask_t __kmpc_impl_lanemask_gt() {
  uint32_t lane = GetLaneId();
  if (lane == (WARPSIZE - 1))
    return 0;
  uint64_t ballot = __kmpc_impl_activemask();
  uint64_t mask = (~((uint64_t)0)) << (lane + 1);
  return mask & ballot;
}

DEVICE double __kmpc_impl_get_wtick() { return ((double)1E-9); }

DEVICE double __kmpc_impl_get_wtime() {
#if __gfx700__ || __gfx701__ || __gfx702__
  uint64_t t = __builtin_amdgcn_s_memtime();
#else
  uint64_t t = __builtin_amdgcn_s_memrealtime();
#endif
  return ((double)1.0 / 745000000.0) * t;
}

// Warp vote function
DEVICE __kmpc_impl_lanemask_t __kmpc_impl_activemask() {
  return __builtin_amdgcn_read_exec();
}

DEVICE int32_t __kmpc_impl_shfl_sync(__kmpc_impl_lanemask_t, int32_t var,
                                     int32_t srcLane) {
  int width = WARPSIZE;
  int self = GetLaneId();
  int index = srcLane + (self & ~(width - 1));
  return __builtin_amdgcn_ds_bpermute(index << 2, var);
}

DEVICE int32_t __kmpc_impl_shfl_down_sync(__kmpc_impl_lanemask_t, int32_t var,
                                          uint32_t laneDelta, int32_t width) {
  int self = GetLaneId();
  int index = self + laneDelta;
  index = (int)(laneDelta + (self & (width - 1))) >= width ? self : index;
  return __builtin_amdgcn_ds_bpermute(index << 2, var);
}

DEVICE int GetNumberOfBlocksInKernel() { return __kmpc_amd_grid_dim_x(); }
DEVICE int GetNumberOfThreadsInBlock() { return __kmpc_amd_workgroup_dim_x(); }
DEVICE unsigned GetWarpId() { return GetThreadIdInBlock() / WARPSIZE; }
DEVICE unsigned GetLaneId() {
  return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}

DEVICE bool __kmpc_impl_is_first_active_thread() {
  __kmpc_impl_lanemask_t Mask = __kmpc_impl_activemask();
  unsigned tid = GetThreadIdInBlock();
  unsigned laneid = (tid & 0X3F);
  int64_t Sh;
  unsigned Shnum;
  if (laneid < 32) {
    Mask = Mask << 32;
    Shnum = 32 - laneid;
  } else {
    Shnum = 64 - laneid;
  }
  Sh = Mask << Shnum;
  return (Sh == (unsigned long long)0);
}

// global_allocate is hostrpc service to return pointer to device memory
// In the future, we will implement a heap allocation and minimize
// host service calls. 
EXTERN char * global_allocate(uint32_t bufsz);
EXTERN int global_free(void*);

// Memory
DEVICE void *__kmpc_impl_malloc(size_t t) { return global_allocate(t); }

DEVICE void __kmpc_impl_free(void * ptr) {global_free(ptr);}
