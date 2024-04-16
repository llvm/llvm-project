//===- ConstantRangeList.cpp - ConstantRangeList implementation -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Represent a list of signed ConstantRange and do NOT support wrap around the
// end of the numeric range. Ranges in the list are ordered and no overlapping.
// Ranges should have the same bitwidth. Each range's lower should be less than
// its upper. Special lists (take 8-bit as an example):
//
// {[0, 0)}     = Empty set
// {[255, 255)} = Full Set
//
// For EmptySet or FullSet, the list size is 1 but it's not allowed to access
// the range.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/ConstantRangeList.h"
#include <cstddef>

using namespace llvm;

ConstantRangeList::ConstantRangeList(uint32_t BitWidth, bool Full) {
  APInt Lower =
      Full ? APInt::getMaxValue(BitWidth) : APInt::getMinValue(BitWidth);
  Ranges.push_back(ConstantRange(Lower, Lower));
}

void ConstantRangeList::insert(const ConstantRange &NewRange) {
  assert(NewRange.getLower().slt(NewRange.getUpper()));
  assert(getBitWidth() == NewRange.getBitWidth());
  // Handle common cases.
  if (isFullSet())
    return;
  if (isEmptySet()) {
    Ranges[0] = NewRange;
    return;
  }
  if (Ranges.back().getUpper().slt(NewRange.getLower())) {
    Ranges.push_back(NewRange);
    return;
  }
  if (NewRange.getUpper().slt(Ranges.front().getLower())) {
    Ranges.insert(Ranges.begin(), NewRange);
    return;
  }

  // Slow insert.
  SmallVector<ConstantRange, 2> ExistingRanges(Ranges.begin(), Ranges.end());
  auto LowerBound =
      std::lower_bound(ExistingRanges.begin(), ExistingRanges.end(), NewRange,
                       [](const ConstantRange &a, const ConstantRange &b) {
                         return a.getLower().slt(b.getLower());
                       });
  Ranges.erase(Ranges.begin() + (LowerBound - ExistingRanges.begin()),
               Ranges.end());
  if (!Ranges.empty() && NewRange.getLower().slt(Ranges.back().getUpper())) {
    APInt NewLower = Ranges.back().getLower();
    APInt NewUpper =
        APIntOps::smax(NewRange.getUpper(), Ranges.back().getUpper());
    Ranges.back() = ConstantRange(NewLower, NewUpper);
  } else {
    Ranges.push_back(NewRange);
  }
  for (auto Iter = LowerBound; Iter != ExistingRanges.end(); Iter++) {
    if (Ranges.back().getUpper().slt(Iter->getLower())) {
      Ranges.push_back(*Iter);
    } else {
      APInt NewLower = Ranges.back().getLower();
      APInt NewUpper =
          APIntOps::smax(Iter->getUpper(), Ranges.back().getUpper());
      Ranges.back() = ConstantRange(NewLower, NewUpper);
    }
  }
  return;
}

void ConstantRangeList::print(raw_ostream &OS) const {
  if (isFullSet())
    OS << "full-set";
  else if (isEmptySet())
    OS << "empty-set";
  else
    for (const auto &Range : Ranges)
      Range.print(OS);
}
