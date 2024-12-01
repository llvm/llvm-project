//===-- Shared/Utils.h - Target independent OpenMP target RTL -- C++ ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Routines and classes used to provide useful functionalities for the host and
// the device.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_SHARED_UTILS_H
#define OMPTARGET_SHARED_UTILS_H

#include <stdint.h>

namespace utils {

/// Return the difference (in bytes) between \p Begin and \p End.
template <typename Ty = char>
auto getPtrDiff(const void *End, const void *Begin) {
  return reinterpret_cast<const Ty *>(End) -
         reinterpret_cast<const Ty *>(Begin);
}

/// Return \p Ptr advanced by \p Offset bytes.
template <typename Ty1, typename Ty2> Ty1 *advancePtr(Ty1 *Ptr, Ty2 Offset) {
  return (Ty1 *)(const_cast<char *>((const char *)(Ptr)) + Offset);
}

/// Return \p V aligned "upwards" according to \p Align.
template <typename Ty1, typename Ty2> inline Ty1 alignPtr(Ty1 V, Ty2 Align) {
  return reinterpret_cast<Ty1>(((uintptr_t(V) + Align - 1) / Align) * Align);
}
/// Return \p V aligned "downwards" according to \p Align.
template <typename Ty1, typename Ty2> inline Ty1 alignDown(Ty1 V, Ty2 Align) {
  return V - V % Align;
}

/// Round up \p V to a \p Boundary.
template <typename Ty> inline Ty roundUp(Ty V, Ty Boundary) {
  return alignPtr(V, Boundary);
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

template <typename DstTy, typename SrcTy> inline DstTy convertViaPun(SrcTy V) {
  static_assert(sizeof(DstTy) == sizeof(SrcTy), "Bad conversion");
  return *((DstTy *)(&V));
}

} // namespace utils

#endif // OMPTARGET_SHARED_UTILS_H
