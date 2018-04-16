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
  JITSymbolResolverAdapter(ExecutionSession &ES, SymbolResolver &R);
  Expected<LookupFlagsResult> lookupFlags(const LookupSet &Symbols) override;
  Expected<LookupResult> lookup(const LookupSet &Symbols) override;

private:
  ExecutionSession &ES;
  std::set<SymbolStringPtr> ResolvedStrings;
  SymbolResolver &R;
};

/// @brief Use the given legacy-style FindSymbol function (i.e. a function that
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

/// @brief Use the given legacy-style FindSymbol function (i.e. a function that
///        takes a const std::string& or StringRef and returns a JITSymbol) to
///        find the address and flags for each symbol in Symbols and store the
///        result in Query. If any JITSymbol returned by FindSymbol is in an
///        error then Query.notifyFailed(...) is called with that error and the
///        function returns immediately. On success, returns the set of symbols
///        not found.
///
/// Useful for implementing lookup bodies that query legacy resolvers.
template <typename FindSymbolFn>
SymbolNameSet lookupWithLegacyFn(AsynchronousSymbolQuery &Query,
                                 const SymbolNameSet &Symbols,
                                 FindSymbolFn FindSymbol) {
  SymbolNameSet SymbolsNotFound;

  for (auto &S : Symbols) {
    if (JITSymbol Sym = FindSymbol(*S)) {
      if (auto Addr = Sym.getAddress()) {
        Query.resolve(S, JITEvaluatedSymbol(*Addr, Sym.getFlags()));
        Query.finalizeSymbol();
      } else {
        Query.notifyMaterializationFailed(Addr.takeError());
        return SymbolNameSet();
      }
    } else if (auto Err = Sym.takeError()) {
      Query.notifyMaterializationFailed(std::move(Err));
      return SymbolNameSet();
    } else
      SymbolsNotFound.insert(S);
  }

  return SymbolsNotFound;
}

/// @brief An ORC SymbolResolver implementation that uses a legacy
///        findSymbol-like function to perform lookup;
template <typename LegacyLookupFn>
class LegacyLookupFnResolver final : public SymbolResolver {
public:
  using ErrorReporter = std::function<void(Error)>;

  LegacyLookupFnResolver(LegacyLookupFn LegacyLookup, ErrorReporter ReportError)
      : LegacyLookup(std::move(LegacyLookup)),
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
    return lookupWithLegacyFn(*Query, Symbols, LegacyLookup);
  }

private:
  LegacyLookupFn LegacyLookup;
  ErrorReporter ReportError;
};

template <typename LegacyLookupFn>
std::shared_ptr<LegacyLookupFnResolver<LegacyLookupFn>>
createLegacyLookupResolver(LegacyLookupFn LegacyLookup,
                           std::function<void(Error)> ErrorReporter) {
  return std::make_shared<LegacyLookupFnResolver<LegacyLookupFn>>(
      std::move(LegacyLookup), std::move(ErrorReporter));
}

} // End namespace orc
} // End namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_LEGACY_H
