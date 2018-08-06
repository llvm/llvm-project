//===- NonRelocatableStringpool.cpp - A simple stringpool  ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "NonRelocatableStringpool.h"

namespace llvm {
namespace dsymutil {

DwarfStringPoolEntryRef NonRelocatableStringpool::getEntry(StringRef S) {
  if (S.empty() && !Strings.empty())
    return EmptyString;

  auto I = Strings.insert({S, DwarfStringPoolEntry()});
  auto &Entry = I.first->second;
  if (I.second || Entry.Index == -1U) {
    Entry.Index = NumEntries++;
    Entry.Offset = CurrentEndOffset;
    Entry.Symbol = nullptr;
    CurrentEndOffset += S.size() + 1;
  }
  return DwarfStringPoolEntryRef(*I.first);
}

StringRef NonRelocatableStringpool::internString(StringRef S) {
  DwarfStringPoolEntry Entry{nullptr, 0, -1U};
  auto InsertResult = Strings.insert({S, Entry});
  return InsertResult.first->getKey();
}

std::vector<DwarfStringPoolEntryRef>
NonRelocatableStringpool::getEntries() const {
  std::vector<DwarfStringPoolEntryRef> Result;
  Result.reserve(Strings.size());
  for (const auto &E : Strings)
    Result.emplace_back(E);
  llvm::sort(
      Result.begin(), Result.end(),
      [](const DwarfStringPoolEntryRef A, const DwarfStringPoolEntryRef B) {
        return A.getIndex() < B.getIndex();
      });
  return Result;
}

} // namespace dsymutil
} // namespace llvm
