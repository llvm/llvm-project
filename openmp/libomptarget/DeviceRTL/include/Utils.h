//===--------- Utils.h - OpenMP device runtime utility functions -- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_DEVICERTL_UTILS_H
#define OMPTARGET_DEVICERTL_UTILS_H

#include "Types.h"

#pragma omp begin declare target device_type(nohost)

namespace _OMP {
namespace utils {

/// Return the value \p Var from thread Id \p SrcLane in the warp if the thread
/// is identified by \p Mask.
int32_t shuffle(uint64_t Mask, int32_t Var, int32_t SrcLane);

int32_t shuffleDown(uint64_t Mask, int32_t Var, uint32_t Delta, int32_t Width);

/// Return \p LowBits and \p HighBits packed into a single 64 bit value.
uint64_t pack(uint32_t LowBits, uint32_t HighBits);

/// Unpack \p Val into \p LowBits and \p HighBits.
void unpack(uint64_t Val, uint32_t &LowBits, uint32_t &HighBits);

/// Round up \p V to a \p Boundary.
template <typename Ty> inline Ty roundUp(Ty V, Ty Boundary) {
  return (V + Boundary - 1) / Boundary * Boundary;
}

/// Advance \p Ptr by \p Bytes bytes.
template <typename Ty1, typename Ty2> inline Ty1 *advance(Ty1 Ptr, Ty2 Bytes) {
  return reinterpret_cast<Ty1 *>(reinterpret_cast<char *>(Ptr) + Bytes);
}

/// Return the first bit set in \p V.
inline uint32_t ffs(uint32_t V) {
  static_assert(sizeof(int) == sizeof(uint32_t), "type size mismatch");
  return __builtin_ffs(V);
}

/// Return the first bit set in \p V.
inline uint32_t ffs(uint64_t V) {
  static_assert(sizeof(long) == sizeof(uint64_t), "type size mismatch");
  return __builtin_ffsl(V);
}

/// Return the number of bits set in \p V.
inline uint32_t popc(uint32_t V) {
  static_assert(sizeof(int) == sizeof(uint32_t), "type size mismatch");
  return __builtin_popcount(V);
}

/// Return the number of bits set in \p V.
inline uint32_t popc(uint64_t V) {
  static_assert(sizeof(long) == sizeof(uint64_t), "type size mismatch");
  return __builtin_popcountl(V);
}

/// Return \p V aligned "upwards" according to \p Align.
template <typename Ty1, typename Ty2> inline Ty1 align_up(Ty1 V, Ty2 Align) {
  return ((V + Ty1(Align) - 1) / Ty1(Align)) * Ty1(Align);
}
/// Return \p V aligned "downwards" according to \p Align.
template <typename Ty1, typename Ty2> inline Ty1 align_down(Ty1 V, Ty2 Align) {
  return V - V % Align;
}

/// Return true iff \p Ptr is pointing into shared (local) memory (AS(3)).
bool isSharedMemPtr(void *Ptr);

/// Return \p V typed punned as \p DstTy.
template <typename DstTy, typename SrcTy> inline DstTy convertViaPun(SrcTy V) {
  return *((DstTy *)(&V));
}

/// A  pointer variable that has by design an `undef` value. Use with care.
__attribute__((loader_uninitialized)) static void *const UndefPtr;

#define OMP_LIKELY(EXPR) __builtin_expect((bool)(EXPR), true)
#define OMP_UNLIKELY(EXPR) __builtin_expect((bool)(EXPR), false)

} // namespace utils
} // namespace _OMP

#pragma omp end declare target

#endif
