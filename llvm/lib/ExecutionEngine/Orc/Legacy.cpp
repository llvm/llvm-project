//===------- Legacy.cpp - Adapters for ExecutionEngine API interop --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Legacy.h"

namespace llvm {
namespace orc {

JITSymbolResolverAdapter::JITSymbolResolverAdapter(
    ExecutionSession &ES, SymbolResolver &R, MaterializationResponsibility *MR)
    : ES(ES), R(R), MR(MR) {}

Expected<JITSymbolResolverAdapter::LookupResult>
JITSymbolResolverAdapter::lookup(const LookupSet &Symbols) {
  SymbolNameSet InternedSymbols;
  for (auto &S : Symbols)
    InternedSymbols.insert(ES.getSymbolStringPool().intern(S));

  auto LookupFn = [&, this](std::shared_ptr<AsynchronousSymbolQuery> Q,
                            SymbolNameSet Unresolved) {
    return R.lookup(std::move(Q), std::move(Unresolved));
  };

  auto InternedResult = blockingLookup(ES, std::move(LookupFn),
                                       std::move(InternedSymbols), false, MR);

  if (!InternedResult)
    return InternedResult.takeError();

  JITSymbolResolver::LookupResult Result;
  for (auto &KV : *InternedResult)
    Result[*KV.first] = KV.second;

  return Result;
}

Expected<JITSymbolResolverAdapter::LookupFlagsResult>
JITSymbolResolverAdapter::lookupFlags(const LookupSet &Symbols) {
  SymbolNameSet InternedSymbols;
  for (auto &S : Symbols)
    InternedSymbols.insert(ES.getSymbolStringPool().intern(S));

  SymbolFlagsMap SymbolFlags;
  R.lookupFlags(SymbolFlags, InternedSymbols);
  LookupFlagsResult Result;
  for (auto &KV : SymbolFlags) {
    ResolvedStrings.insert(KV.first);
    Result[*KV.first] = KV.second;
  }

  return Result;
}

} // End namespace orc.
} // End namespace llvm.
