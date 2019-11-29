//===------- Legacy.cpp - Adapters for ExecutionEngine API interop --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Legacy.h"

namespace llvm {
namespace orc {

void SymbolResolver::anchor() {}

JITSymbolResolverAdapter::JITSymbolResolverAdapter(
    ExecutionSession &ES, SymbolResolver &R, MaterializationResponsibility *MR)
    : ES(ES), R(R), MR(MR) {}

void JITSymbolResolverAdapter::lookup(const LookupSet &Symbols,
                                      OnResolvedFunction OnResolved) {
  SymbolNameSet InternedSymbols;
  for (auto &S : Symbols)
    InternedSymbols.insert(ES.intern(S));

  auto OnResolvedWithUnwrap = [OnResolved = std::move(OnResolved)](
                                  Expected<SymbolMap> InternedResult) mutable {
    if (!InternedResult) {
      OnResolved(InternedResult.takeError());
      return;
    }

    LookupResult Result;
    for (auto &KV : *InternedResult)
      Result[*KV.first] = std::move(KV.second);
    OnResolved(Result);
  };

  auto Q = std::make_shared<AsynchronousSymbolQuery>(
      SymbolLookupSet(InternedSymbols), SymbolState::Resolved,
      std::move(OnResolvedWithUnwrap));

  auto Unresolved = R.lookup(Q, InternedSymbols);
  if (Unresolved.empty()) {
    if (MR)
      MR->addDependenciesForAll(Q->QueryRegistrations);
  } else
    ES.legacyFailQuery(*Q, make_error<SymbolsNotFound>(std::move(Unresolved)));
}

Expected<JITSymbolResolverAdapter::LookupSet>
JITSymbolResolverAdapter::getResponsibilitySet(const LookupSet &Symbols) {
  SymbolNameSet InternedSymbols;
  for (auto &S : Symbols)
    InternedSymbols.insert(ES.intern(S));

  auto InternedResult = R.getResponsibilitySet(InternedSymbols);
  LookupSet Result;
  for (auto &S : InternedResult) {
    ResolvedStrings.insert(S);
    Result.insert(*S);
  }

  return Result;
}

} // End namespace orc.
} // End namespace llvm.
