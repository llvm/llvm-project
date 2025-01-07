//===--- DeviceUtils.h - OpenMP device runtime utility functions -- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_DEVICERTL_DEVICE_UTILS_H
#define OMPTARGET_DEVICERTL_DEVICE_UTILS_H

#include "DeviceTypes.h"
#include "Shared/Utils.h"

#pragma omp begin declare target device_type(nohost)

namespace utils {

/// Return the value \p Var from thread Id \p SrcLane in the warp if the thread
/// is identified by \p Mask.
int32_t shuffle(uint64_t Mask, int32_t Var, int32_t SrcLane, int32_t Width);

int32_t shuffleDown(uint64_t Mask, int32_t Var, uint32_t Delta, int32_t Width);

int64_t shuffleDown(uint64_t Mask, int64_t Var, uint32_t Delta, int32_t Width);

uint64_t ballotSync(uint64_t Mask, int32_t Pred);

/// Return \p LowBits and \p HighBits packed into a single 64 bit value.
uint64_t pack(uint32_t LowBits, uint32_t HighBits);

/// Unpack \p Val into \p LowBits and \p HighBits.
void unpack(uint64_t Val, uint32_t &LowBits, uint32_t &HighBits);

/// Return true iff \p Ptr is pointing into shared (local) memory (AS(3)).
bool isSharedMemPtr(void *Ptr);

/// Return true iff \p Ptr is pointing into (thread) local memory (AS(5)).
bool isThreadLocalMemPtr(void *Ptr);

/// A  pointer variable that has by design an `undef` value. Use with care.
[[clang::loader_uninitialized]] static void *const UndefPtr;

#define OMP_LIKELY(EXPR) __builtin_expect((bool)(EXPR), true)
#define OMP_UNLIKELY(EXPR) __builtin_expect((bool)(EXPR), false)

} // namespace utils

#pragma omp end declare target

#endif
