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

void ConstantRangeList::insert(const ConstantRange &Range) {
  assert(Range.getLower().slt(Range.getUpper()));
  assert(getBitWidth() == Range.getBitWidth());
  if (isFullSet())
    return;
  if (isEmptySet()) {
    Ranges[0] = Range;
    return;
  }

  ConstantRange RangeToInsert = Range;
  SmallVector<ConstantRange, 2> ExistingRanges(Ranges.begin(), Ranges.end());
  Ranges.clear();
  for (size_t i = 0; i < ExistingRanges.size(); i++) {
    const ConstantRange &CurRange = ExistingRanges[i];
    if (CurRange.getUpper().slt(RangeToInsert.getLower())) {
      // Case1: No overlap and CurRange is before ToInsert.
      // |--CurRange--|
      //                 |--ToInsert--|
      Ranges.push_back(CurRange);
      continue;
    } else if (RangeToInsert.getUpper().slt(CurRange.getLower())) {
      // Case2: No overlap and CurRange is after ToInsert.
      //                                 |--CurRange--|
      //                 |--ToInsert--|
      // insert the range.
      Ranges.push_back(RangeToInsert);
      for (size_t j = i; j < ExistingRanges.size(); j++)
        Ranges.push_back(ExistingRanges[j]);
      return;
    } else {
      // Case3: Overlap.
      APInt NewLower =
          APIntOps::smin(CurRange.getLower(), RangeToInsert.getLower());
      APInt NewUpper =
          APIntOps::smax(CurRange.getUpper(), RangeToInsert.getUpper());
      RangeToInsert = ConstantRange(NewLower, NewUpper);
    }
  }
  Ranges.push_back(RangeToInsert);
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
