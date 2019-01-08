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

  if (Translator)
    S = Translator(S);
  auto I = Strings.insert({S, DwarfStringPoolEntry()});
  auto &Entry = I.first->second;
  if (I.second || !Entry.isIndexed()) {
    Entry.Index = NumEntries++;
    Entry.Offset = CurrentEndOffset;
    Entry.Symbol = nullptr;
    CurrentEndOffset += S.size() + 1;
  }
  return DwarfStringPoolEntryRef(*I.first, true);
}

StringRef NonRelocatableStringpool::internString(StringRef S) {
  DwarfStringPoolEntry Entry{nullptr, 0, DwarfStringPoolEntry::NotIndexed};

  if (Translator)
    S = Translator(S);

  auto InsertResult = Strings.insert({S, Entry});
  return InsertResult.first->getKey();
}

std::vector<DwarfStringPoolEntryRef>
NonRelocatableStringpool::getEntriesForEmission() const {
  std::vector<DwarfStringPoolEntryRef> Result;
  Result.reserve(Strings.size());
  for (const auto &E : Strings)
    if (E.getValue().isIndexed())
      Result.emplace_back(E, true);
  llvm::sort(Result, [](const DwarfStringPoolEntryRef A,
                        const DwarfStringPoolEntryRef B) {
    return A.getIndex() < B.getIndex();
  });
  return Result;
}

} // namespace dsymutil
} // namespace llvm
