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

AddressRanges::Collection::const_iterator
AddressRanges::insert(AddressRange Range) {
  if (Range.size() == 0)
    return Ranges.end();

  auto It = llvm::upper_bound(Ranges, Range);
  auto It2 = It;
  while (It2 != Ranges.end() && It2->start() <= Range.end())
    ++It2;
  if (It != It2) {
    Range = {Range.start(), std::max(Range.end(), std::prev(It2)->end())};
    It = Ranges.erase(It, It2);
  }
  if (It != Ranges.begin() && Range.start() <= std::prev(It)->end()) {
    --It;
    *It = {It->start(), std::max(It->end(), Range.end())};
    return It;
  }

  return Ranges.insert(It, Range);
}

AddressRanges::Collection::const_iterator
AddressRanges::find(uint64_t Addr) const {
  auto It = std::partition_point(
      Ranges.begin(), Ranges.end(),
      [=](const AddressRange &R) { return R.start() <= Addr; });

  if (It == Ranges.begin())
    return Ranges.end();

  --It;
  if (Addr >= It->end())
    return Ranges.end();

  return It;
}

AddressRanges::Collection::const_iterator
AddressRanges::find(AddressRange Range) const {
  if (Range.size() == 0)
    return Ranges.end();

  auto It = std::partition_point(
      Ranges.begin(), Ranges.end(),
      [=](const AddressRange &R) { return R.start() <= Range.start(); });

  if (It == Ranges.begin())
    return Ranges.end();

  --It;
  if (Range.end() > It->end())
    return Ranges.end();

  return It;
}
