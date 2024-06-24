//===- ConstantRangeList.cpp - ConstantRangeList implementation -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/ConstantRangeList.h"
#include <cstddef>

using namespace llvm;

bool ConstantRangeList::isOrderedRanges(ArrayRef<ConstantRange> RangesRef) {
  if (RangesRef.empty())
    return true;
  auto Range = RangesRef[0];
  if (Range.getLower().sge(Range.getUpper()))
    return false;
  for (unsigned i = 1; i < RangesRef.size(); i++) {
    auto CurRange = RangesRef[i];
    auto PreRange = RangesRef[i - 1];
    if (CurRange.getLower().sge(CurRange.getUpper()) ||
        CurRange.getLower().sle(PreRange.getUpper()))
      return false;
  }
  return true;
}

std::optional<ConstantRangeList>
ConstantRangeList::getConstantRangeList(ArrayRef<ConstantRange> RangesRef) {
  if (!isOrderedRanges(RangesRef))
    return std::nullopt;
  return ConstantRangeList(RangesRef);
}

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

  auto LowerBound = lower_bound(
      Ranges, NewRange, [](const ConstantRange &a, const ConstantRange &b) {
        return a.getLower().slt(b.getLower());
      });
  if (LowerBound != Ranges.end() && LowerBound->contains(NewRange))
    return;

  // Slow insert.
  SmallVector<ConstantRange, 2> ExistingTail(LowerBound, Ranges.end());
  Ranges.erase(LowerBound, Ranges.end());
  // Merge consecutive ranges.
  if (!Ranges.empty() && NewRange.getLower().sle(Ranges.back().getUpper())) {
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

ConstantRangeList
ConstantRangeList::unionWith(const ConstantRangeList &CRL) const {
  assert(getBitWidth() == CRL.getBitWidth() &&
         "ConstantRangeList types don't agree!");
  // Handle common cases.
  if (empty())
    return CRL;
  if (CRL.empty())
    return *this;

  ConstantRangeList Result;
  size_t i = 0, j = 0;
  ConstantRange PreviousRange(getBitWidth(), false);
  if (Ranges[i].getLower().slt(CRL.Ranges[j].getLower())) {
    PreviousRange = Ranges[i++];
  } else {
    PreviousRange = CRL.Ranges[j++];
  }
  auto UnionAndUpdateRange = [&PreviousRange,
                              &Result](const ConstantRange &CR) {
    assert(!CR.isSignWrappedSet() && "Upper wrapped ranges are not supported");
    if (PreviousRange.getUpper().slt(CR.getLower())) {
      Result.Ranges.push_back(PreviousRange);
      PreviousRange = CR;
    } else {
      PreviousRange = ConstantRange(
          PreviousRange.getLower(),
          APIntOps::smax(PreviousRange.getUpper(), CR.getUpper()));
    }
  };
  while (i < size() || j < CRL.size()) {
    if (j == CRL.size() ||
        (i < size() && Ranges[i].getLower().slt(CRL.Ranges[j].getLower()))) {
      // Merge PreviousRange with this.
      UnionAndUpdateRange(Ranges[i++]);
    } else {
      // Merge PreviousRange with CRL.
      UnionAndUpdateRange(CRL.Ranges[j++]);
    }
  }
  Result.Ranges.push_back(PreviousRange);
  return Result;
}

ConstantRangeList
ConstantRangeList::intersectWith(const ConstantRangeList &CRL) const {
  assert(getBitWidth() == CRL.getBitWidth() &&
         "ConstantRangeList types don't agree!");

  // Handle common cases.
  if (empty())
    return *this;
  if (CRL.empty())
    return CRL;

  ConstantRangeList Result;
  size_t i = 0, j = 0;
  while (i < size() && j < CRL.size()) {
    auto &Range = this->Ranges[i];
    auto &OtherRange = CRL.Ranges[j];
    assert(!Range.isSignWrappedSet() && !OtherRange.isSignWrappedSet() &&
           "Upper wrapped ranges are not supported");

    APInt Start = Range.getLower().slt(OtherRange.getLower())
                      ? OtherRange.getLower()
                      : Range.getLower();
    APInt End = Range.getUpper().slt(OtherRange.getUpper())
                    ? Range.getUpper()
                    : OtherRange.getUpper();
    if (Start.slt(End))
      Result.Ranges.push_back(ConstantRange(Start, End));

    if (Range.getUpper().slt(OtherRange.getUpper()))
      i++;
    else
      j++;
  }
  return Result;
}

void ConstantRangeList::print(raw_ostream &OS) const {
  interleaveComma(Ranges, OS, [&](ConstantRange CR) {
    OS << "(" << CR.getLower() << ", " << CR.getUpper() << ")";
  });
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void ConstantRangeList::dump() const {
  print(dbgs());
  dbgs() << '\n';
}
#endif
