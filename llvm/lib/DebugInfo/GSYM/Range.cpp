//===- Range.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/GSYM/Range.h"
#include <algorithm>
#include <inttypes.h>

using namespace llvm;
using namespace gsym;


void AddressRanges::insert(AddressRange Range) {
  if (Range.size() == 0)
    return;

  auto It = llvm::upper_bound(Ranges, Range);
  auto It2 = It;
  while (It2 != Ranges.end() && It2->Start < Range.End)
    ++It2;
  if (It != It2) {
    Range.End = std::max(Range.End, It2[-1].End);
    It = Ranges.erase(It, It2);
  }
  if (It != Ranges.begin() && Range.Start < It[-1].End)
    It[-1].End = std::max(It[-1].End, Range.End);
  else
    Ranges.insert(It, Range);
}

bool AddressRanges::contains(uint64_t Addr) const {
  auto It = std::partition_point(
      Ranges.begin(), Ranges.end(),
      [=](const AddressRange &R) { return R.Start <= Addr; });
  return It != Ranges.begin() && Addr < It[-1].End;
}

raw_ostream &llvm::gsym::operator<<(raw_ostream &OS, const AddressRange &R) {
  return OS << '[' << HEX64(R.Start) << " - " << HEX64(R.End) << ")";
}

raw_ostream &llvm::gsym::operator<<(raw_ostream &OS, const AddressRanges &AR) {
  size_t Size = AR.size();
  for (size_t I = 0; I < Size; ++I) {
    if (I)
      OS << ' ';
    OS << AR[I];
  }
  return OS;
}
