//===------- Utils.cpp - OpenMP device runtime utility functions -- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "DeviceUtils.h"

#include "Debug.h"
#include "Interface.h"
#include "Mapping.h"

#pragma omp begin declare target device_type(nohost)

using namespace ompx;

namespace impl {

bool isSharedMemPtr(const void *Ptr) { return false; }

void Unpack(uint64_t Val, uint32_t *LowBits, uint32_t *HighBits) {
  static_assert(sizeof(unsigned long) == 8, "");
  *LowBits = static_cast<uint32_t>(Val & 0x00000000FFFFFFFFUL);
  *HighBits = static_cast<uint32_t>((Val & 0xFFFFFFFF00000000UL) >> 32);
}

uint64_t Pack(uint32_t LowBits, uint32_t HighBits) {
  return (((uint64_t)HighBits) << 32) | (uint64_t)LowBits;
}

int32_t shuffle(uint64_t Mask, int32_t Var, int32_t SrcLane, int32_t Width);
int32_t shuffleDown(uint64_t Mask, int32_t Var, uint32_t LaneDelta,
                    int32_t Width);

uint64_t ballotSync(uint64_t Mask, int32_t Pred);

/// AMDGCN Implementation
///
///{
#pragma omp begin declare variant match(device = {arch(amdgcn)})

int32_t shuffle(uint64_t Mask, int32_t Var, int32_t SrcLane, int32_t Width) {
  int Self = mapping::getThreadIdInWarp();
  int Index = SrcLane + (Self & ~(Width - 1));
  return __builtin_amdgcn_ds_bpermute(Index << 2, Var);
}

int32_t shuffleDown(uint64_t Mask, int32_t Var, uint32_t LaneDelta,
                    int32_t Width) {
  int Self = mapping::getThreadIdInWarp();
  int Index = Self + LaneDelta;
  Index = (int)(LaneDelta + (Self & (Width - 1))) >= Width ? Self : Index;
  return __builtin_amdgcn_ds_bpermute(Index << 2, Var);
}

uint64_t ballotSync(uint64_t Mask, int32_t Pred) {
  return Mask & __builtin_amdgcn_ballot_w64(Pred);
}

bool isSharedMemPtr(const void *Ptr) {
  return __builtin_amdgcn_is_shared(
      (const __attribute__((address_space(0))) void *)Ptr);
}
#pragma omp end declare variant
///}

/// NVPTX Implementation
///
///{
#pragma omp begin declare variant match(                                       \
        device = {arch(nvptx, nvptx64)},                                       \
            implementation = {extension(match_any)})

int32_t shuffle(uint64_t Mask, int32_t Var, int32_t SrcLane, int32_t Width) {
  return __nvvm_shfl_sync_idx_i32(Mask, Var, SrcLane, Width - 1);
}

int32_t shuffleDown(uint64_t Mask, int32_t Var, uint32_t Delta, int32_t Width) {
  int32_t T = ((mapping::getWarpSize() - Width) << 8) | 0x1f;
  return __nvvm_shfl_sync_down_i32(Mask, Var, Delta, T);
}

uint64_t ballotSync(uint64_t Mask, int32_t Pred) {
  return __nvvm_vote_ballot_sync(static_cast<uint32_t>(Mask), Pred);
}

bool isSharedMemPtr(const void *Ptr) { return __nvvm_isspacep_shared(Ptr); }

#pragma omp end declare variant
///}
} // namespace impl

uint64_t utils::pack(uint32_t LowBits, uint32_t HighBits) {
  return impl::Pack(LowBits, HighBits);
}

void utils::unpack(uint64_t Val, uint32_t &LowBits, uint32_t &HighBits) {
  impl::Unpack(Val, &LowBits, &HighBits);
}

int32_t utils::shuffle(uint64_t Mask, int32_t Var, int32_t SrcLane,
                       int32_t Width) {
  return impl::shuffle(Mask, Var, SrcLane, Width);
}

int32_t utils::shuffleDown(uint64_t Mask, int32_t Var, uint32_t Delta,
                           int32_t Width) {
  return impl::shuffleDown(Mask, Var, Delta, Width);
}

int64_t utils::shuffleDown(uint64_t Mask, int64_t Var, uint32_t Delta,
                           int32_t Width) {
  uint32_t Lo, Hi;
  utils::unpack(Var, Lo, Hi);
  Hi = impl::shuffleDown(Mask, Hi, Delta, Width);
  Lo = impl::shuffleDown(Mask, Lo, Delta, Width);
  return utils::pack(Lo, Hi);
}

uint64_t utils::ballotSync(uint64_t Mask, int32_t Pred) {
  return impl::ballotSync(Mask, Pred);
}

bool utils::isSharedMemPtr(void *Ptr) { return impl::isSharedMemPtr(Ptr); }

extern "C" {
int32_t __kmpc_shuffle_int32(int32_t Val, int16_t Delta, int16_t SrcLane) {
  return impl::shuffleDown(lanes::All, Val, Delta, SrcLane);
}

int64_t __kmpc_shuffle_int64(int64_t Val, int16_t Delta, int16_t Width) {
  return utils::shuffleDown(lanes::All, Val, Delta, Width);
}
}

#pragma omp end declare target
