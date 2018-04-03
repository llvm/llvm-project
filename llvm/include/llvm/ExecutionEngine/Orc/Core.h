//===------ Core.h -- Core ORC APIs (Layer, JITDylib, etc.) -----*- C++ -*-===//
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

#ifndef LLVM_EXECUTIONENGINE_ORC_CORE_H
#define LLVM_EXECUTIONENGINE_ORC_CORE_H

#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/SymbolStringPool.h"

#include <map>
#include <memory>
#include <set>
#include <vector>

namespace llvm {
namespace orc {

/// VModuleKey provides a unique identifier (allocated and managed by
/// ExecutionSessions) for a module added to the JIT.
using VModuleKey = uint64_t;

class VSO;

/// @brief A set of symbol names (represented by SymbolStringPtrs for
//         efficiency).
using SymbolNameSet = std::set<SymbolStringPtr>;

/// @brief A map from symbol names (as SymbolStringPtrs) to JITSymbols
///        (address/flags pairs).
using SymbolMap = std::map<SymbolStringPtr, JITEvaluatedSymbol>;

/// @brief A map from symbol names (as SymbolStringPtrs) to JITSymbolFlags.
using SymbolFlagsMap = std::map<SymbolStringPtr, JITSymbolFlags>;

/// @brief A symbol query that returns results via a callback when results are
///        ready.
///
/// makes a callback when all symbols are available.
class AsynchronousSymbolQuery {
public:
  /// @brief Callback to notify client that symbols have been resolved.
  using SymbolsResolvedCallback = std::function<void(Expected<SymbolMap>)>;

  /// @brief Callback to notify client that symbols are ready for execution.
  using SymbolsReadyCallback = std::function<void(Error)>;

  /// @brief Create a query for the given symbols, notify-resolved and
  ///        notify-ready callbacks.
  AsynchronousSymbolQuery(const SymbolNameSet &Symbols,
                          SymbolsResolvedCallback NotifySymbolsResolved,
                          SymbolsReadyCallback NotifySymbolsReady);

  /// @brief Notify client that the query failed.
  ///
  /// If the notify-resolved callback has not been made yet, then it is called
  /// with the given error, and the notify-finalized callback is never made.
  ///
  /// If the notify-resolved callback has already been made then then the
  /// notify-finalized callback is called with the given error.
  ///
  /// It is illegal to call setFailed after both callbacks have been made.
  void setFailed(Error Err);

  /// @brief Set the resolved symbol information for the given symbol name.
  ///
  /// If this symbol was the last one not resolved, this will trigger a call to
  /// the notify-finalized callback passing the completed sybol map.
  void setDefinition(SymbolStringPtr Name, JITEvaluatedSymbol Sym);

  /// @brief Notify the query that a requested symbol is ready for execution.
  ///
  /// This decrements the query's internal count of not-yet-ready symbols. If
  /// this call to notifySymbolFinalized sets the counter to zero, it will call
  /// the notify-finalized callback with Error::success as the value.
  void notifySymbolFinalized();

private:
  SymbolMap Symbols;
  size_t OutstandingResolutions = 0;
  size_t OutstandingFinalizations = 0;
  SymbolsResolvedCallback NotifySymbolsResolved;
  SymbolsReadyCallback NotifySymbolsReady;
};

/// @brief SymbolResolver is a composable interface for looking up symbol flags
///        and addresses using the AsynchronousSymbolQuery type. It will
///        eventually replace the LegacyJITSymbolResolver interface as the
///        stardard ORC symbol resolver type.
class SymbolResolver {
public:
  virtual ~SymbolResolver() = default;

  /// @brief Returns the flags for each symbol in Symbols that can be found,
  ///        along with the set of symbol that could not be found.
  virtual SymbolNameSet lookupFlags(SymbolFlagsMap &Flags,
                                    const SymbolNameSet &Symbols) = 0;

  /// @brief For each symbol in Symbols that can be found, assigns that symbols
  ///        value in Query. Returns the set of symbols that could not be found.
  virtual SymbolNameSet lookup(std::shared_ptr<AsynchronousSymbolQuery> Query,
                               SymbolNameSet Symbols) = 0;

private:
  virtual void anchor();
};

/// @brief Implements SymbolResolver with a pair of supplied function objects
///        for convenience. See createSymbolResolver.
template <typename LookupFlagsFn, typename LookupFn>
class LambdaSymbolResolver final : public SymbolResolver {
public:
  template <typename LookupFlagsFnRef, typename LookupFnRef>
  LambdaSymbolResolver(LookupFlagsFnRef &&LookupFlags, LookupFnRef &&Lookup)
      : LookupFlags(std::forward<LookupFlagsFnRef>(LookupFlags)),
        Lookup(std::forward<LookupFnRef>(Lookup)) {}

  SymbolNameSet lookupFlags(SymbolFlagsMap &Flags,
                            const SymbolNameSet &Symbols) final {
    return LookupFlags(Flags, Symbols);
  }

  SymbolNameSet lookup(std::shared_ptr<AsynchronousSymbolQuery> Query,
                       SymbolNameSet Symbols) final {
    return Lookup(std::move(Query), std::move(Symbols));
  }

private:
  LookupFlagsFn LookupFlags;
  LookupFn Lookup;
};

/// @brief Creates a SymbolResolver implementation from the pair of supplied
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

/// @brief A MaterializationUnit represents a set of symbol definitions that can
///        be materialized as a group, or individually discarded (when
///        overriding definitions are encountered).
///
/// MaterializationUnits are used when providing lazy definitions of symbols to
/// VSOs. The VSO will call materialize when the address of a symbol is
/// requested via the lookup method. The VSO will call discard if a stronger
/// definition is added or already present.
class MaterializationUnit {
public:
  virtual ~MaterializationUnit() {}

  /// @brief Return the set of symbols that this source provides.
  virtual SymbolFlagsMap getSymbols() = 0;

  /// @brief Implementations of this method should materialize all symbols
  ///        in the materialzation unit, except for those that have been
  ///        previously discarded.
  virtual Error materialize(VSO &V) = 0;

  /// @brief Implementations of this method should discard the given symbol
  ///        from the source (e.g. if the source is an LLVM IR Module and the
  ///        symbol is a function, delete the function body or mark it available
  ///        externally).
  virtual void discard(VSO &V, SymbolStringPtr Name) = 0;

private:
  virtual void anchor();
};

/// @brief Represents a dynamic linkage unit in a JIT process.
///
/// VSO acts as a symbol table (symbol definitions can be set and the dylib
/// queried to find symbol addresses) and as a key for tracking resources
/// (since a VSO's address is fixed).
class VSO {
  friend class ExecutionSession;

public:
  enum RelativeLinkageStrength {
    NewDefinitionIsStronger,
    DuplicateDefinition,
    ExistingDefinitionIsStronger
  };

  using SetDefinitionsResult =
      std::map<SymbolStringPtr, RelativeLinkageStrength>;

  using MaterializationUnitList =
      std::vector<std::unique_ptr<MaterializationUnit>>;

  struct LookupResult {
    MaterializationUnitList MaterializationUnits;
    SymbolNameSet UnresolvedSymbols;
  };

  VSO() = default;

  VSO(const VSO &) = delete;
  VSO &operator=(const VSO &) = delete;
  VSO(VSO &&) = delete;
  VSO &operator=(VSO &&) = delete;

  /// @brief Compare new linkage with existing linkage.
  static RelativeLinkageStrength
  compareLinkage(Optional<JITSymbolFlags> OldFlags, JITSymbolFlags NewFlags);

  /// @brief Compare new linkage with an existing symbol's linkage.
  RelativeLinkageStrength compareLinkage(SymbolStringPtr Name,
                                         JITSymbolFlags NewFlags) const;

  /// @brief Adds the given symbols to the mapping as resolved, finalized
  ///        symbols.
  ///
  /// FIXME: We can take this by const-ref once symbol-based laziness is
  ///        removed.
  Error define(SymbolMap NewSymbols);

  /// @brief Adds the given symbols to the mapping as lazy symbols.
  Error defineLazy(std::unique_ptr<MaterializationUnit> Source);

  /// @brief Add the given symbol/address mappings to the dylib, but do not
  ///        mark the symbols as finalized yet.
  void resolve(SymbolMap SymbolValues);

  /// @brief Finalize the given symbols.
  void finalize(SymbolNameSet SymbolsToFinalize);

  /// @brief Look up the flags for the given symbols.
  ///
  /// Returns the flags for the give symbols, together with the set of symbols
  /// not found.
  SymbolNameSet lookupFlags(SymbolFlagsMap &Flags, SymbolNameSet Symbols);

  /// @brief Apply the given query to the given symbols in this VSO.
  ///
  /// For symbols in this VSO that have already been materialized, their address
  /// will be set in the query immediately.
  ///
  /// For symbols in this VSO that have not been materialized, the query will be
  /// recorded and the source for those symbols (plus the set of symbols to be
  /// materialized by that source) will be returned as the MaterializationWork
  /// field of the LookupResult.
  ///
  /// Any symbols not found in this VSO will be returned in the
  /// UnresolvedSymbols field of the LookupResult.
  LookupResult lookup(std::shared_ptr<AsynchronousSymbolQuery> Query,
                      SymbolNameSet Symbols);

private:
  class MaterializationInfo {
  public:
    using QueryList = std::vector<std::shared_ptr<AsynchronousSymbolQuery>>;

    MaterializationInfo(size_t SymbolsRemaining,
                        std::unique_ptr<MaterializationUnit> MU);

    uint64_t SymbolsRemaining;
    std::unique_ptr<MaterializationUnit> MU;
    SymbolMap Symbols;
    std::map<SymbolStringPtr, QueryList> PendingResolution;
    std::map<SymbolStringPtr, QueryList> PendingFinalization;
  };

  using MaterializationInfoSet = std::set<std::unique_ptr<MaterializationInfo>>;

  using MaterializationInfoIterator = MaterializationInfoSet::iterator;

  class SymbolTableEntry {
  public:
    SymbolTableEntry(JITSymbolFlags SymbolFlags,
                     MaterializationInfoIterator MaterializationInfoItr);
    SymbolTableEntry(JITEvaluatedSymbol Sym);
    SymbolTableEntry(SymbolTableEntry &&Other);
    ~SymbolTableEntry();

    SymbolTableEntry &operator=(JITEvaluatedSymbol Sym);

    JITSymbolFlags getFlags() const;
    void replaceWith(VSO &V, SymbolStringPtr Name, JITSymbolFlags Flags,
                     MaterializationInfoIterator NewMaterializationInfoItr);
    std::unique_ptr<MaterializationUnit>
    query(SymbolStringPtr Name, std::shared_ptr<AsynchronousSymbolQuery> Query);
    void resolve(VSO &V, SymbolStringPtr Name, JITEvaluatedSymbol Sym);
    void finalize(VSO &V, SymbolStringPtr Name);
    void discard(VSO &V, SymbolStringPtr Name);

  private:
    void destroy();

    JITSymbolFlags Flags;
    MaterializationInfoIterator MII;
    union {
      JITTargetAddress Address;
      MaterializationInfoIterator MaterializationInfoItr;
    };
  };

  std::map<SymbolStringPtr, SymbolTableEntry> Symbols;
  MaterializationInfoSet MaterializationInfos;
};

/// @brief An ExecutionSession represents a running JIT program.
class ExecutionSession {
public:
  using ErrorReporter = std::function<void(Error)>;

  /// @brief Construct an ExecutionEngine.
  ///
  /// SymbolStringPools may be shared between ExecutionSessions.
  ExecutionSession(std::shared_ptr<SymbolStringPool> SSP = nullptr)
    : SSP(SSP ? std::move(SSP) : std::make_shared<SymbolStringPool>()) {}

  /// @brief Returns the SymbolStringPool for this ExecutionSession.
  SymbolStringPool &getSymbolStringPool() const { return *SSP; }

  /// @brief Set the error reporter function.
  void setErrorReporter(ErrorReporter ReportError) {
    this->ReportError = std::move(ReportError);
  }

  /// @brief Report a error for this execution session.
  ///
  /// Unhandled errors can be sent here to log them.
  void reportError(Error Err) { ReportError(std::move(Err)); }

  /// @brief Allocate a module key for a new module to add to the JIT.
  VModuleKey allocateVModule() { return ++LastKey; }

  /// @brief Return a module key to the ExecutionSession so that it can be
  ///        re-used. This should only be done once all resources associated
  ////       with the original key have been released.
  void releaseVModule(VModuleKey Key) { /* FIXME: Recycle keys */ }

public:
  static void logErrorsToStdErr(Error Err);

  std::shared_ptr<SymbolStringPool> SSP;
  VModuleKey LastKey = 0;
  ErrorReporter ReportError = logErrorsToStdErr;
};

/// Runs Materializers on the current thread and reports errors to the given
/// ExecutionSession.
class MaterializeOnCurrentThread {
public:
  MaterializeOnCurrentThread(ExecutionSession &ES) : ES(ES) {}

  void operator()(VSO &V, std::unique_ptr<MaterializationUnit> MU) {
    if (auto Err = MU->materialize(V))
      ES.reportError(std::move(Err));
  }

private:
  ExecutionSession &ES;
};

/// Materialization function object wrapper for the lookup method.
using MaterializationDispatcher =
    std::function<void(VSO &V, std::unique_ptr<MaterializationUnit> S)>;

/// @brief Look up a set of symbols by searching a list of VSOs.
///
/// All VSOs in the list should be non-null.
Expected<SymbolMap> lookup(const std::vector<VSO *> &VSOs, SymbolNameSet Names,
                           MaterializationDispatcher DispatchMaterialization);

/// @brief Look up a symbol by searching a list of VSOs.
Expected<JITEvaluatedSymbol>
lookup(const std::vector<VSO *> VSOs, SymbolStringPtr Name,
       MaterializationDispatcher DispatchMaterialization);

} // End namespace orc
} // End namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_CORE_H
