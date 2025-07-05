//===- HLSLRootSignatureValidations.cpp - HLSL Root Signature helpers -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains helpers for working with HLSL Root Signatures.
///
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/HLSL/RootSignatureValidations.h"

namespace llvm {
namespace hlsl {
namespace rootsig {

std::optional<const RangeInfo *>
ResourceRange::getOverlapping(const RangeInfo &Info) const {
  MapT::const_iterator Interval = Intervals.find(Info.LowerBound);
  if (!Interval.valid() || Info.UpperBound < Interval.start())
    return std::nullopt;
  return Interval.value();
}

const RangeInfo *ResourceRange::lookup(uint32_t X) const {
  return Intervals.lookup(X, nullptr);
}

void ResourceRange::clear() { return Intervals.clear(); }

std::optional<const RangeInfo *> ResourceRange::insert(const RangeInfo &Info) {
  uint32_t LowerBound = Info.LowerBound;
  uint32_t UpperBound = Info.UpperBound;

  std::optional<const RangeInfo *> Res = std::nullopt;
  MapT::iterator Interval = Intervals.begin();

  while (true) {
    if (UpperBound < LowerBound)
      break;

    Interval.advanceTo(LowerBound);
    if (!Interval.valid()) // No interval found
      break;

    // Let Interval = [x;y] and [LowerBound;UpperBound] = [a;b] and note that
    // a <= y implicitly from Intervals.find(LowerBound)
    if (UpperBound < Interval.start())
      break; // found interval does not overlap with inserted one

    if (!Res.has_value()) // Update to be the first found intersection
      Res = Interval.value();

    if (Interval.start() <= LowerBound && UpperBound <= Interval.stop()) {
      // x <= a <= b <= y implies that [a;b] is covered by [x;y]
      //  -> so we don't need to insert this, report an overlap
      return Res;
    } else if (LowerBound <= Interval.start() &&
               Interval.stop() <= UpperBound) {
      // a <= x <= y <= b implies that [x;y] is covered by [a;b]
      //  -> so remove the existing interval that we will cover with the
      //  overwrite
      Interval.erase();
    } else if (LowerBound < Interval.start() && UpperBound <= Interval.stop()) {
      // a < x <= b <= y implies that [a; x] is not covered but [x;b] is
      //  -> so set b = x - 1 such that [a;x-1] is now the interval to insert
      UpperBound = Interval.start() - 1;
    } else if (Interval.start() <= LowerBound && Interval.stop() < UpperBound) {
      // a < x <= b <= y implies that [y; b] is not covered but [a;y] is
      //  -> so set a = y + 1 such that [y+1;b] is now the interval to insert
      LowerBound = Interval.stop() + 1;
    }
  }

  assert(LowerBound <= UpperBound && "Attempting to insert an empty interval");
  Intervals.insert(LowerBound, UpperBound, &Info);
  return Res;
}

} // namespace rootsig
} // namespace hlsl
} // namespace llvm
