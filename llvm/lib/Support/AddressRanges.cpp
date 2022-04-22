//===- AddressRanges.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/AddressRanges.h"
#include "llvm/ADT/STLExtras.h"
#include <inttypes.h>

using namespace llvm;

void AddressRanges::insert(AddressRange Range) {
  if (Range.size() == 0)
    return;

  auto It = llvm::upper_bound(Ranges, Range);
  auto It2 = It;
  while (It2 != Ranges.end() && It2->start() < Range.end())
    ++It2;
  if (It != It2) {
    Range = {Range.start(), std::max(Range.end(), It2[-1].end())};
    It = Ranges.erase(It, It2);
  }
  if (It != Ranges.begin() && Range.start() < It[-1].end())
    It[-1] = {It[-1].start(), std::max(It[-1].end(), Range.end())};
  else
    Ranges.insert(It, Range);
}

bool AddressRanges::contains(uint64_t Addr) const {
  auto It = std::partition_point(
      Ranges.begin(), Ranges.end(),
      [=](const AddressRange &R) { return R.start() <= Addr; });
  return It != Ranges.begin() && Addr < It[-1].end();
}

bool AddressRanges::contains(AddressRange Range) const {
  if (Range.size() == 0)
    return false;
  auto It = std::partition_point(
      Ranges.begin(), Ranges.end(),
      [=](const AddressRange &R) { return R.start() <= Range.start(); });
  if (It == Ranges.begin())
    return false;
  return Range.end() <= It[-1].end();
}

Optional<AddressRange>
AddressRanges::getRangeThatContains(uint64_t Addr) const {
  auto It = std::partition_point(
      Ranges.begin(), Ranges.end(),
      [=](const AddressRange &R) { return R.start() <= Addr; });
  if (It != Ranges.begin() && Addr < It[-1].end())
    return It[-1];
  return llvm::None;
}
