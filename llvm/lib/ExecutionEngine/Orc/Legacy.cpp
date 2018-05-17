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
  Error Err = Error::success();
  JITSymbolResolver::LookupResult Result;

  SymbolNameSet InternedSymbols;
  for (auto &S : Symbols)
    InternedSymbols.insert(ES.getSymbolStringPool().intern(S));

  auto OnResolve =
      [&, this](Expected<AsynchronousSymbolQuery::ResolutionResult> RR) {
        if (RR) {
          // If this lookup was attached to a MaterializationResponsibility then
          // record the dependencies.
          if (MR)
            MR->addDependencies(RR->Dependencies);

          for (auto &KV : RR->Symbols) {
            ResolvedStrings.insert(KV.first);
            Result[*KV.first] = KV.second;
          }
        } else
          Err = joinErrors(std::move(Err), RR.takeError());
      };

  auto OnReady = [this](Error Err) { ES.reportError(std::move(Err)); };

  auto Query = std::make_shared<AsynchronousSymbolQuery>(InternedSymbols,
                                                         OnResolve, OnReady);

  auto UnresolvedSymbols = R.lookup(Query, InternedSymbols);

  if (!UnresolvedSymbols.empty()) {
    std::string ErrorMsg = "Unresolved symbols: ";

    ErrorMsg += **UnresolvedSymbols.begin();
    for (auto I = std::next(UnresolvedSymbols.begin()),
              E = UnresolvedSymbols.end();
         I != E; ++I) {
      ErrorMsg += ", ";
      ErrorMsg += **I;
    }

    Err =
        joinErrors(std::move(Err),
                   make_error<StringError>(ErrorMsg, inconvertibleErrorCode()));
  }

  if (Err)
    return std::move(Err);

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
