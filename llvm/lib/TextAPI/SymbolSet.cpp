//===- lib/TextAPI/SymbolSet.cpp - TAPI Symbol Set ------------*- C++-*----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/TextAPI/SymbolSet.h"

using namespace llvm;
using namespace llvm::MachO;

Symbol *SymbolSet::addGlobalImpl(EncodeKind Kind, StringRef Name,
                                 SymbolFlags Flags) {
  Name = copyString(Name);
  auto Result = Symbols.try_emplace(SymbolsMapKey{Kind, Name}, nullptr);
  if (Result.second)
    Result.first->second =
        new (Allocator) Symbol{Kind, Name, TargetList(), Flags};
  return Result.first->second;
}

Symbol *SymbolSet::addGlobal(EncodeKind Kind, StringRef Name, SymbolFlags Flags,
                             const Target &Targ) {
  auto *Sym = addGlobalImpl(Kind, Name, Flags);
  Sym->addTarget(Targ);
  return Sym;
}

const Symbol *SymbolSet::findSymbol(EncodeKind Kind, StringRef Name) const {
  return Symbols.lookup({Kind, Name});
}
