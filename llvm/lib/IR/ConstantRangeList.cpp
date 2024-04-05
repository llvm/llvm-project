//===- ConstantRangeList.cpp - ConstantRangeList implementation -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Represent a list of signed ConstantRange and do NOT support wrap around the
// end of the numeric range. Ranges in the list should have the same bitwidth.
// Each range's lower should be less than its upper. Special lists (take 8-bit
// as an example):
//
// {[0, 0)}     = Empty set
// {[255, 255)} = Full Set
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

ConstantRangeList::ConstantRangeList(int64_t Lower, int64_t Upper) {
  Ranges.push_back(
      ConstantRange(APInt(64, StringRef(std::to_string(Lower)), 10),
                    APInt(64, StringRef(std::to_string(Upper)), 10)));
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
