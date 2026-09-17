//===- LoopInterchangeUtils.h - Numeric helpers for LoopInterchange -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// IR-free numeric helpers for LoopInterchange's outer-epilogue fission:
// checked magnitudes, lossless unsigned comparisons, and byte-range
// containment.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TRANSFORMS_SCALAR_LOOPINTERCHANGEUTILS_H
#define LLVM_LIB_TRANSFORMS_SCALAR_LOOPINTERCHANGEUTILS_H

#include "llvm/ADT/APInt.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

namespace llvm {
namespace loop_interchange_utils {

/// Unsigned magnitude of \p Value, or nullopt above INT64_MAX, so that the
/// result and its negation both fit in int64_t.
inline std::optional<uint64_t> checkedAbsToUnsigned(int64_t Value) {
  uint64_t Magnitude = AbsoluteValue(Value);
  if (Magnitude > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    return std::nullopt;
  return Magnitude;
}

/// Zero-extend both APInts to the larger bit width, preserving all bits.
inline std::pair<APInt, APInt> zeroExtendToCommonWidth(const APInt &A,
                                                       const APInt &B) {
  unsigned Width = std::max(A.getBitWidth(), B.getBitWidth());
  return {A.zext(Width), B.zext(Width)};
}

/// True iff \p A u<= \p B after lossless zero-extension to a common width.
inline bool unsignedLEWithZeroExtend(const APInt &A, const APInt &B) {
  std::pair<APInt, APInt> Widened = zeroExtendToCommonWidth(A, B);
  return Widened.first.ule(Widened.second);
}

/// Check containment of a fixed byte range in a known object.
/// Given a closed byte range of element start offsets
/// [\p FirstByte, \p LastByteStart] and a fixed \p ElementSize, verify the
/// whole accessed range
/// [\p FirstByte, \p LastByteStart + ElementSize - 1] stays inside
/// [0, \p ObjectSize). Rejects a negative start, a reversed range, an element
/// size of zero, an element size that cannot be represented as a signed byte
/// offset, signed overflow while adding ElementSize - 1 to the last element
/// start, and a final byte at or past ObjectSize.
inline bool isFixedByteRangeWithinObject(int64_t FirstByte,
                                         int64_t LastByteStart,
                                         uint64_t ElementSize,
                                         uint64_t ObjectSize) {
  if (FirstByte < 0 || LastByteStart < FirstByte || ElementSize == 0 ||
      ElementSize > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    return false;
  int64_t LastByte;
  if (AddOverflow(LastByteStart, static_cast<int64_t>(ElementSize) - 1,
                  LastByte))
    return false;
  return LastByte >= 0 && static_cast<uint64_t>(LastByte) < ObjectSize;
}

} // namespace loop_interchange_utils
} // namespace llvm

#endif // LLVM_LIB_TRANSFORMS_SCALAR_LOOPINTERCHANGEUTILS_H
