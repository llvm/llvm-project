//===---------- AbsoluteSymbols.cpp - Absolute symbols utilities ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/AbsoluteSymbols.h"
#include "llvm/ExecutionEngine/Orc/Core.h"

#define DEBUG_TYPE "orc"

namespace llvm::orc {

AbsoluteSymbolsMaterializationUnit::AbsoluteSymbolsMaterializationUnit(
    SymbolMap Symbols)
    : MaterializationUnit(extractFlags(Symbols)), Symbols(std::move(Symbols)) {}

StringRef AbsoluteSymbolsMaterializationUnit::getName() const {
  return "<Absolute Symbols>";
}

void AbsoluteSymbolsMaterializationUnit::materialize(
    std::unique_ptr<MaterializationResponsibility> R) {
  // Even though these are just absolute symbols we need to check for failure
  // to resolve/emit: the tracker for these symbols may have been removed while
  // the materialization was in flight (e.g. due to a failure in some action
  // triggered by the queries attached to the resolution/emission of these
  // symbols).
  if (auto Err = R->notifyResolved(Symbols)) {
    R->getExecutionSession().reportError(std::move(Err));
    R->failMaterialization();
    return;
  }
  if (auto Err = R->notifyEmitted({})) {
    R->getExecutionSession().reportError(std::move(Err));
    R->failMaterialization();
    return;
  }
}

void AbsoluteSymbolsMaterializationUnit::discard(const JITDylib &JD,
                                                 const SymbolStringPtr &Name) {
  assert(Symbols.count(Name) && "Symbol is not part of this MU");
  Symbols.erase(Name);
}

MaterializationUnit::Interface
AbsoluteSymbolsMaterializationUnit::extractFlags(const SymbolMap &Symbols) {
  SymbolFlagsMap Flags;
  for (const auto &[Name, Def] : Symbols)
    Flags[Name] = Def.getFlags();
  return MaterializationUnit::Interface(std::move(Flags), nullptr);
}

} // namespace llvm::orc
