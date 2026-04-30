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

} // namespace utils

#endif // OMPTARGET_SHARED_UTILS_H
