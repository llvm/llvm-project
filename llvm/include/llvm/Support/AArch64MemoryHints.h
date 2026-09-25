//===-- AArch64MemoryHints.h - AArch64 Memory Hint Attributes ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_AARCH64MEMORYHINTS_H
#define LLVM_SUPPORT_AARCH64MEMORYHINTS_H

namespace llvm {
enum class AArch64MemoryHint {
  NONE = 0,
  STSHH_KEEP = 1,
  STSHH_STRM = 2,
};

template <typename Int> inline bool isValidAArch64MemoryHintValue(Int I) {
  return (Int)AArch64MemoryHint::STSHH_KEEP <= I &&
         I <= (Int)AArch64MemoryHint::STSHH_STRM;
}

template <typename Int> inline AArch64MemoryHint toAArch64MemoryHint(Int I) {
  switch (I) {
  case 0:
    return AArch64MemoryHint::STSHH_KEEP;
  case 1:
    return AArch64MemoryHint::STSHH_STRM;
  default:
    return AArch64MemoryHint::NONE;
  }
}
} // namespace llvm
#endif // LLVM_SUPPORT_AARCH64MEMORYHINTS_H
