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

/// SymbolResolver is a composable interface for looking up symbol flags
///        and addresses using the AsynchronousSymbolQuery type. It will
///        eventually replace the LegacyJITSymbolResolver interface as the
///        stardard ORC symbol resolver type.
///
/// FIXME: SymbolResolvers should go away and be replaced with VSOs with
///        defenition generators.
class SymbolResolver {
public:
  virtual ~SymbolResolver() = default;

  /// Returns the flags for each symbol in Symbols that can be found,
  ///        along with the set of symbol that could not be found.
  virtual SymbolFlagsMap lookupFlags(const SymbolNameSet &Symbols) = 0;

  /// For each symbol in Symbols that can be found, assigns that symbols
  ///        value in Query. Returns the set of symbols that could not be found.
  virtual SymbolNameSet lookup(std::shared_ptr<AsynchronousSymbolQuery> Query,
                               SymbolNameSet Symbols) = 0;

private:
  virtual void anchor();
};

/// Implements SymbolResolver with a pair of supplied function objects
///        for convenience. See createSymbolResolver.
template <typename LookupFlagsFn, typename LookupFn>
class LambdaSymbolResolver final : public SymbolResolver {
public:
  template <typename LookupFlagsFnRef, typename LookupFnRef>
  LambdaSymbolResolver(LookupFlagsFnRef &&LookupFlags, LookupFnRef &&Lookup)
      : LookupFlags(std::forward<LookupFlagsFnRef>(LookupFlags)),
        Lookup(std::forward<LookupFnRef>(Lookup)) {}

  SymbolFlagsMap lookupFlags(const SymbolNameSet &Symbols) final {
    return LookupFlags(Symbols);
  }

  SymbolNameSet lookup(std::shared_ptr<AsynchronousSymbolQuery> Query,
                       SymbolNameSet Symbols) final {
    return Lookup(std::move(Query), std::move(Symbols));
  }

private:
  LookupFlagsFn LookupFlags;
  LookupFn Lookup;
};

/// Creates a SymbolResolver implementation from the pair of supplied
///        function objects.
template <typename LookupFlagsFn, typename LookupFn>
std::unique_ptr<LambdaSymbolResolver<
    typename std::remove_cv<
        typename std::remove_reference<LookupFlagsFn>::type>::type,
    typename std::remove_cv<
        typename std::remove_reference<LookupFn>::type>::type>>
createSymbolResolver(LookupFlagsFn &&LookupFlags, LookupFn &&Lookup) {
  using LambdaSymbolResolverImpl = LambdaSymbolResolver<
      typename std::remove_cv<
          typename std::remove_reference<LookupFlagsFn>::type>::type,
      typename std::remove_cv<
          typename std::remove_reference<LookupFn>::type>::type>;
  return llvm::make_unique<LambdaSymbolResolverImpl>(
      std::forward<LookupFlagsFn>(LookupFlags), std::forward<LookupFn>(Lookup));
}

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
Expected<SymbolFlagsMap> lookupFlagsWithLegacyFn(const SymbolNameSet &Symbols,
                                                 FindSymbolFn FindSymbol) {
  SymbolFlagsMap SymbolFlags;

  for (auto &S : Symbols) {
    if (JITSymbol Sym = FindSymbol(*S))
      SymbolFlags[S] = Sym.getFlags();
    else if (auto Err = Sym.takeError())
      return std::move(Err);
  }

  return SymbolFlags;
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
        ES.legacyFailQuery(Query, Addr.takeError());
        return SymbolNameSet();
      }
    } else if (auto Err = Sym.takeError()) {
      ES.legacyFailQuery(Query, std::move(Err));
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

  SymbolFlagsMap lookupFlags(const SymbolNameSet &Symbols) final {
    if (auto SymbolFlags = lookupFlagsWithLegacyFn(Symbols, LegacyLookup))
      return std::move(*SymbolFlags);
    else {
      ReportError(SymbolFlags.takeError());
      return SymbolFlagsMap();
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
