//===- Range.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/GSYM/Range.h"
#include "llvm/DebugInfo/GSYM/FileWriter.h"
#include "llvm/Support/DataExtractor.h"
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

bool AddressRanges::contains(AddressRange Range) const {
  if (Range.size() == 0)
    return false;
  auto It = std::partition_point(
      Ranges.begin(), Ranges.end(),
      [=](const AddressRange &R) { return R.Start <= Range.Start; });
  if (It == Ranges.begin())
    return false;
  return Range.End <= It[-1].End;
}

Optional<AddressRange>
AddressRanges::getRangeThatContains(uint64_t Addr) const {
  auto It = std::partition_point(
      Ranges.begin(), Ranges.end(),
      [=](const AddressRange &R) { return R.Start <= Addr; });
  if (It != Ranges.begin() && Addr < It[-1].End)
    return It[-1];
  return llvm::None;
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

void AddressRange::encode(FileWriter &O, uint64_t BaseAddr) const {
  assert(Start >= BaseAddr);
  O.writeULEB(Start - BaseAddr);
  O.writeULEB(size());
}

void AddressRange::decode(DataExtractor &Data, uint64_t BaseAddr,
                          uint64_t &Offset) {
  const uint64_t AddrOffset = Data.getULEB128(&Offset);
  const uint64_t Size = Data.getULEB128(&Offset);
  const uint64_t StartAddr = BaseAddr + AddrOffset;
  Start = StartAddr;
  End = StartAddr + Size;
}

void AddressRanges::encode(FileWriter &O, uint64_t BaseAddr) const {
  O.writeULEB(Ranges.size());
  if (Ranges.empty())
    return;
  for (auto Range : Ranges)
    Range.encode(O, BaseAddr);
}

void AddressRanges::decode(DataExtractor &Data, uint64_t BaseAddr,
                           uint64_t &Offset) {
  clear();
  uint64_t NumRanges = Data.getULEB128(&Offset);
  if (NumRanges == 0)
    return;
  Ranges.resize(NumRanges);
  for (auto &Range : Ranges)
    Range.decode(Data, BaseAddr, Offset);
}

void AddressRange::skip(DataExtractor &Data, uint64_t &Offset) {
  Data.getULEB128(&Offset);
  Data.getULEB128(&Offset);
}

uint64_t AddressRanges::skip(DataExtractor &Data, uint64_t &Offset) {
  uint64_t NumRanges = Data.getULEB128(&Offset);
  for (uint64_t I=0; I<NumRanges; ++I)
    AddressRange::skip(Data, Offset);
  return NumRanges;
}
