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
// its upper.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/ConstantRangeList.h"
#include <cstddef>

using namespace llvm;

void ConstantRangeList::insert(const ConstantRange &NewRange) {
  if (NewRange.isEmptySet())
    return;
  assert(!NewRange.isFullSet() && "Do not support full set");
  assert(NewRange.getLower().slt(NewRange.getUpper()));
  assert(getBitWidth() == NewRange.getBitWidth());
  // Handle common cases.
  if (empty() || Ranges.back().getUpper().slt(NewRange.getLower())) {
    Ranges.push_back(NewRange);
    return;
  }
  if (NewRange.getUpper().slt(Ranges.front().getLower())) {
    Ranges.insert(Ranges.begin(), NewRange);
    return;
  }
  if (std::find(Ranges.begin(), Ranges.end(), NewRange) != Ranges.end()) {
    return;
  }

  // Slow insert.
  auto LowerBound =
      std::lower_bound(Ranges.begin(), Ranges.end(), NewRange,
                       [](const ConstantRange &a, const ConstantRange &b) {
                         return a.getLower().slt(b.getLower());
                       });
  SmallVector<ConstantRange, 2> ExistingTail(LowerBound, Ranges.end());
  Ranges.erase(LowerBound, Ranges.end());
  if (!Ranges.empty() && NewRange.getLower().slt(Ranges.back().getUpper())) {
    APInt NewLower = Ranges.back().getLower();
    APInt NewUpper =
        APIntOps::smax(NewRange.getUpper(), Ranges.back().getUpper());
    Ranges.back() = ConstantRange(NewLower, NewUpper);
  } else {
    Ranges.push_back(NewRange);
  }
  for (auto Iter = ExistingTail.begin(); Iter != ExistingTail.end(); Iter++) {
    if (Ranges.back().getUpper().slt(Iter->getLower())) {
      Ranges.push_back(*Iter);
    } else {
      APInt NewLower = Ranges.back().getLower();
      APInt NewUpper =
          APIntOps::smax(Iter->getUpper(), Ranges.back().getUpper());
      Ranges.back() = ConstantRange(NewLower, NewUpper);
    }
  }
}

void ConstantRangeList::print(raw_ostream &OS) const {
  interleaveComma(Ranges, OS, [&](ConstantRange CR) {
    OS << "(" << CR.getLower() << ", " << CR.getUpper() << ")";
  });
}
