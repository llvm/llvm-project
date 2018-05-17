//===--- Legacy.h -- Adapters for ExecutionEngine API interop ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Contains core ORC APIs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_LEGACY_H
#define LLVM_EXECUTIONENGINE_ORC_LEGACY_H

#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/Core.h"

namespace llvm {
namespace orc {

class JITSymbolResolverAdapter : public JITSymbolResolver {
public:
  JITSymbolResolverAdapter(ExecutionSession &ES, SymbolResolver &R,
                           MaterializationResponsibility *MR);
  Expected<LookupFlagsResult> lookupFlags(const LookupSet &Symbols) override;
  Expected<LookupResult> lookup(const LookupSet &Symbols) override;

private:
  ExecutionSession &ES;
  std::set<SymbolStringPtr> ResolvedStrings;
  SymbolResolver &R;
  MaterializationResponsibility *MR;
};

/// Use the given legacy-style FindSymbol function (i.e. a function that
///        takes a const std::string& or StringRef and returns a JITSymbol) to
///        find the flags for each symbol in Symbols and store their flags in
///        SymbolFlags. If any JITSymbol returned by FindSymbol is in an error
///        state the function returns immediately with that error, otherwise it
///        returns the set of symbols not found.
///
/// Useful for implementing lookupFlags bodies that query legacy resolvers.
template <typename FindSymbolFn>
Expected<SymbolNameSet> lookupFlagsWithLegacyFn(SymbolFlagsMap &SymbolFlags,
                                                const SymbolNameSet &Symbols,
                                                FindSymbolFn FindSymbol) {
  SymbolNameSet SymbolsNotFound;

  for (auto &S : Symbols) {
    if (JITSymbol Sym = FindSymbol(*S))
      SymbolFlags[S] = Sym.getFlags();
    else if (auto Err = Sym.takeError())
      return std::move(Err);
    else
      SymbolsNotFound.insert(S);
  }

  return SymbolsNotFound;
}

/// Use the given legacy-style FindSymbol function (i.e. a function that
///        takes a const std::string& or StringRef and returns a JITSymbol) to
///        find the address and flags for each symbol in Symbols and store the
///        result in Query. If any JITSymbol returned by FindSymbol is in an
///        error then Query.notifyFailed(...) is called with that error and the
///        function returns immediately. On success, returns the set of symbols
///        not found.
///
/// Useful for implementing lookup bodies that query legacy resolvers.
template <typename FindSymbolFn>
SymbolNameSet
lookupWithLegacyFn(ExecutionSession &ES, AsynchronousSymbolQuery &Query,
                   const SymbolNameSet &Symbols, FindSymbolFn FindSymbol) {
  SymbolNameSet SymbolsNotFound;
  bool NewSymbolsResolved = false;

  for (auto &S : Symbols) {
    if (JITSymbol Sym = FindSymbol(*S)) {
      if (auto Addr = Sym.getAddress()) {
        Query.resolve(S, JITEvaluatedSymbol(*Addr, Sym.getFlags()));
        Query.notifySymbolReady();
        NewSymbolsResolved = true;
      } else {
        ES.failQuery(Query, Addr.takeError());
        return SymbolNameSet();
      }
    } else if (auto Err = Sym.takeError()) {
      ES.failQuery(Query, std::move(Err));
      return SymbolNameSet();
    } else
      SymbolsNotFound.insert(S);
  }

  if (NewSymbolsResolved && Query.isFullyResolved())
    Query.handleFullyResolved();

  if (NewSymbolsResolved && Query.isFullyReady())
    Query.handleFullyReady();

  return SymbolsNotFound;
}

/// An ORC SymbolResolver implementation that uses a legacy
///        findSymbol-like function to perform lookup;
template <typename LegacyLookupFn>
class LegacyLookupFnResolver final : public SymbolResolver {
public:
  using ErrorReporter = std::function<void(Error)>;

  LegacyLookupFnResolver(ExecutionSession &ES, LegacyLookupFn LegacyLookup,
                         ErrorReporter ReportError)
      : ES(ES), LegacyLookup(std::move(LegacyLookup)),
        ReportError(std::move(ReportError)) {}

  SymbolNameSet lookupFlags(SymbolFlagsMap &Flags,
                            const SymbolNameSet &Symbols) final {
    if (auto RemainingSymbols =
            lookupFlagsWithLegacyFn(Flags, Symbols, LegacyLookup))
      return std::move(*RemainingSymbols);
    else {
      ReportError(RemainingSymbols.takeError());
      return Symbols;
    }
  }

  SymbolNameSet lookup(std::shared_ptr<AsynchronousSymbolQuery> Query,
                       SymbolNameSet Symbols) final {
    return lookupWithLegacyFn(ES, *Query, Symbols, LegacyLookup);
  }

private:
  ExecutionSession &ES;
  LegacyLookupFn LegacyLookup;
  ErrorReporter ReportError;
};

template <typename LegacyLookupFn>
std::shared_ptr<LegacyLookupFnResolver<LegacyLookupFn>>
createLegacyLookupResolver(ExecutionSession &ES, LegacyLookupFn LegacyLookup,
                           std::function<void(Error)> ErrorReporter) {
  return std::make_shared<LegacyLookupFnResolver<LegacyLookupFn>>(
      ES, std::move(LegacyLookup), std::move(ErrorReporter));
}

} // End namespace orc
} // End namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_LEGACY_H
