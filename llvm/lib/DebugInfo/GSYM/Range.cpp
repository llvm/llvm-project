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


void AddressRanges::insert(const AddressRange &Range) {
  if (Range.size() == 0)
    return;
  // Ranges.insert(std::upper_bound(Ranges.begin(), Ranges.end(), Range), Range);

  // // Check if an existing range intersects with this range, and if so, 
  // // grow the intersecting ranges instead of adding a new one.
  auto Begin = Ranges.begin();
  auto End = Ranges.end();
  const auto Iter = std::upper_bound(Begin, End, Range);
  if (Iter != Begin) {
    auto PrevIter = Iter - 1;
    // If the previous range itersects with "Range" they will be combined.
    if (PrevIter->intersect(Range)) {
      // Now check if the previous range intersects with the next range since
      // the previous range was combined. If so, combine them and remove the
      // next range.
      if (Iter != End && PrevIter->intersect(*Iter))
        Ranges.erase(Iter);
      return;
    }
  }
  // If the next range intersects with "Range", combined and return.
  if (Iter != End && Iter->intersect(Range))
    return;
  Ranges.insert(Iter, Range);
}

bool AddressRanges::contains(uint64_t Addr) const {
  auto It = std::partition_point(
      Ranges.begin(), Ranges.end(),
      [=](const AddressRange &R) { return R.startAddress() <= Addr; });
  return It != Ranges.begin() && It[-1].contains(Addr);
}

raw_ostream &llvm::gsym::operator<<(raw_ostream &OS, const AddressRange &R) {
  return OS << '[' << HEX64(R.startAddress()) << " - " << HEX64(R.endAddress())
            << ")";
}

raw_ostream &llvm::gsym::operator<<(raw_ostream &OS, const AddressRanges &AR) {
  size_t Size = AR.size();
  for (size_t I=0; I<Size; ++I) {
    if (I)
      OS << ' ';
    OS << AR[I];
  }
  return OS;
}

