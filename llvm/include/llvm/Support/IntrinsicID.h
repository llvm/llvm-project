//===- llvm/Support/IntrinsicID.h - Intrinsic ID encoding -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains functions to support intrinsic ID encoding. The
// Intrinsic::ID enum value is constructed using a target prefix index in bits
// 23-16 (8-bit) and an intrinsic index (index within the list of intrinsics for
// tha target) in lower 16 bits. To support Intrinsic::ID 0 being not used, the
// intrinsic index is encoded as Index + 1 for all targets.
//
// This file defines functions that encapsulate this encoding.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_INTRINSIC_ID_H
#define LLVM_SUPPORT_INTRINSIC_ID_H

#include <limits>
#include <optional>
#include <utility>

namespace llvm::Intrinsic {
typedef unsigned ID;

inline ID encodeIntrinsicID(unsigned TargetIndex, unsigned IntrinsicIndex) {
  assert(IntrinsicIndex < std::numeric_limits<uint16_t>::max());
  assert(TargetIndex <= std::numeric_limits<uint8_t>::max());
  return (TargetIndex << 16) | (IntrinsicIndex + 1);
}

inline std::pair<unsigned, unsigned> decodeIntrinsicID(ID id) {
  unsigned IntrinsicIndex = id & 0xFFFF;
  unsigned TargetIndex = id >> 16;
  assert(IntrinsicIndex != 0);
  return {TargetIndex, IntrinsicIndex - 1};
}

inline std::optional<std::pair<unsigned, unsigned>>
decodeIntrinsicIDNoFail(ID id) {
  unsigned IntrinsicIndex = id & 0xFFFF;
  unsigned TargetIndex = id >> 16;
  if (IntrinsicIndex == 0)
    return std::nullopt;
  return std::make_pair(TargetIndex, IntrinsicIndex - 1);
}

} // end namespace llvm::Intrinsic

#endif // LLVM_SUPPORT_INTRINSIC_ID_H
