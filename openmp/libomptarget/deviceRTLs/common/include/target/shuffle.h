//===- shuffle.h - OpenMP variants of the shuffle idiom for all targets -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shuffle function implementations for all supported targets.
//
// Note: We unify the mask type to uint64_t instead of __kmpc_impl_lanemask_t.
//
//===----------------------------------------------------------------------===//

#ifndef LIBOMPTARGET_DEVICERTL_SHUFFLE_H
#define LIBOMPTARGET_DEVICERTL_SHUFFLE_H

#include <assert.h>
#include <inttypes.h>

#pragma omp declare target

/// External shuffle API
///
///{

extern "C" {
int32_t __kmpc_shuffle_int32(int32_t val, int16_t delta, int16_t size);
int64_t __kmpc_shuffle_int64(int64_t val, int16_t delta, int16_t size);
}

///}

/// Forward declarations
///
///{
unsigned GetLaneId();
unsigned GetWarpSize();
void __kmpc_impl_unpack(uint64_t val, uint32_t &lo, uint32_t &hi);
uint64_t __kmpc_impl_pack(uint32_t lo, uint32_t hi);
///}

/// Fallback implementations of the shuffle sync idiom.
///
///{

inline int32_t __kmpc_impl_shfl_sync(uint64_t Mask, int32_t Var,
                                     int32_t SrcLane) {
  assert(false &&
         "Fallback version of __kmpc_impl_shfl_sync is not available!");
}

inline int32_t __kmpc_impl_shfl_down_sync(uint64_t Mask, int32_t Var,
                                          uint32_t Delta, int32_t Width) {
  assert(false &&
         "Fallback version of __kmpc_impl_shfl_down_sync is not available!");
}

///}

/// AMDGCN implementations of the shuffle sync idiom.
///
///{
#pragma omp begin declare variant match(device = {arch(amdgcn)})

inline int32_t __kmpc_impl_shfl_sync(uint64_t Mask, int32_t Var,
                                     int32_t SrcLane) {
  int Width = GetWarpSize();
  int Self = GetLaneId();
  int Index = SrcLane + (Self & ~(Width - 1));
  return __builtin_amdgcn_ds_bpermute(Index << 2, Var);
}

inline int32_t __kmpc_impl_shfl_down_sync(uint64_t Mask, int32_t Var,
                                          uint32_t LaneDelta, int32_t Width) {
  int Self = GetLaneId();
  int Index = Self + LaneDelta;
  Index = (int)(LaneDelta + (Self & (Width - 1))) >= Width ? Self : Index;
  return __builtin_amdgcn_ds_bpermute(Index << 2, Var);
}

#pragma omp end declare variant
///}

/// NVPTX implementations of the shuffle and shuffle sync idiom.
///
///{
#pragma omp begin declare variant match(                                       \
    device = {arch(nvptx, nvptx64)}, implementation = {extension(match_any)})

inline int32_t __kmpc_impl_shfl_sync(uint64_t Mask, int32_t Var,
                                     int32_t SrcLane) {
  return __nvvm_shfl_idx_i32(Var, SrcLane, 0x1f);
}

inline int32_t __kmpc_impl_shfl_down_sync(uint64_t Mask, int32_t Var,
                                          uint32_t Delta, int32_t Width) {
  int32_t T = ((GetWarpSize() - Width) << 8) | 0x1f;
  return __nvvm_shfl_down_i32(Var, Delta, T);
}

#pragma omp end declare variant
///}

#pragma omp end declare target

#endif
