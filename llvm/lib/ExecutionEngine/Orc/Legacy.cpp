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

JITSymbolResolverAdapter::JITSymbolResolverAdapter(ExecutionSession &ES,
                                                   SymbolResolver &R)
    : ES(ES), R(R) {}

Expected<JITSymbolResolverAdapter::LookupResult>
JITSymbolResolverAdapter::lookup(const LookupSet &Symbols) {
  Error Err = Error::success();
  JITSymbolResolver::LookupResult Result;

  SymbolNameSet InternedSymbols;
  for (auto &S : Symbols)
    InternedSymbols.insert(ES.getSymbolStringPool().intern(S));

  auto OnResolve = [&](Expected<SymbolMap> R) {
    if (R) {
      for (auto &KV : *R) {
        ResolvedStrings.insert(KV.first);
        Result[*KV.first] = KV.second;
      }
    } else
      Err = joinErrors(std::move(Err), R.takeError());
  };

  auto OnReady = [](Error Err) {
    // FIXME: Report error to ExecutionSession.
    logAllUnhandledErrors(std::move(Err), errs(),
                          "legacy resolver received on-ready error:\n");
  };

  auto Query = std::make_shared<AsynchronousSymbolQuery>(InternedSymbols,
                                                         OnResolve, OnReady);

  auto UnresolvedSymbols = R.lookup(std::move(Query), InternedSymbols);

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
