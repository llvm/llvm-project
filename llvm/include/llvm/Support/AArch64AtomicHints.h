//===-- AArch64AtomicHints.h - AArch64 Atomic Hint Attributes ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_AARCH64ATOMICHINTS_H
#define LLVM_SUPPORT_AARCH64ATOMICHINTS_H

namespace llvm {
enum class AArch64AtomicStoreHint {
  HINT_NONE = 0,
  HINT_STSHH_KEEP = 1,
  HINT_STSHH_STRM = 2,
};

template <typename Int> inline bool isValidAArch64AtomicHintValue(Int I) {
  return (Int)AArch64AtomicStoreHint::HINT_STSHH_KEEP <= I &&
         I <= (Int)AArch64AtomicStoreHint::HINT_STSHH_STRM;
}

template <typename Int>
inline AArch64AtomicStoreHint getAtomicStoreHintFromMD(Int I) {
  switch (I) {
  case 0:
    return AArch64AtomicStoreHint::HINT_STSHH_KEEP;
  case 1:
    return AArch64AtomicStoreHint::HINT_STSHH_STRM;
  default:
    return AArch64AtomicStoreHint::HINT_NONE;
  }
}
} // namespace llvm
#endif // LLVM_SUPPORT_AARCH64ATOMICHINTS_H
