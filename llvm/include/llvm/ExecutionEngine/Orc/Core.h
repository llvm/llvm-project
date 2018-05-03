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

#include <list>
#include <map>
#include <memory>
#include <set>
#include <vector>

namespace llvm {
namespace orc {

// Forward declare some classes.
class VSO;

/// VModuleKey provides a unique identifier (allocated and managed by
/// ExecutionSessions) for a module added to the JIT.
using VModuleKey = uint64_t;

/// A set of symbol names (represented by SymbolStringPtrs for
//         efficiency).
using SymbolNameSet = std::set<SymbolStringPtr>;

/// Render a SymbolNameSet to an ostream.
raw_ostream &operator<<(raw_ostream &OS, const SymbolNameSet &Symbols);

/// A map from symbol names (as SymbolStringPtrs) to JITSymbols
///        (address/flags pairs).
using SymbolMap = std::map<SymbolStringPtr, JITEvaluatedSymbol>;

/// Render a SymbolMap to an ostream.
raw_ostream &operator<<(raw_ostream &OS, const SymbolMap &Symbols);

/// A map from symbol names (as SymbolStringPtrs) to JITSymbolFlags.
using SymbolFlagsMap = std::map<SymbolStringPtr, JITSymbolFlags>;

/// Render a SymbolMap to an ostream.
raw_ostream &operator<<(raw_ostream &OS, const SymbolFlagsMap &Symbols);

/// A base class for materialization failures that allows the failing
///        symbols to be obtained for logging.
class FailedToMaterialize : public ErrorInfo<FailedToMaterialize> {
public:
  static char ID;
  virtual const SymbolNameSet &getSymbols() const = 0;
};

/// Used to notify a VSO that the given set of symbols failed to resolve.
class FailedToResolve : public ErrorInfo<FailedToResolve, FailedToMaterialize> {
public:
  static char ID;

  FailedToResolve(SymbolNameSet Symbols);
  std::error_code convertToErrorCode() const override;
  void log(raw_ostream &OS) const override;
  const SymbolNameSet &getSymbols() const override { return Symbols; }

private:
  SymbolNameSet Symbols;
};

/// Used to notify a VSO that the given set of symbols failed to
/// finalize.
class FailedToFinalize
    : public ErrorInfo<FailedToFinalize, FailedToMaterialize> {
public:
  static char ID;

  FailedToFinalize(SymbolNameSet Symbols);
  std::error_code convertToErrorCode() const override;
  void log(raw_ostream &OS) const override;
  const SymbolNameSet &getSymbols() const override { return Symbols; }

private:
  SymbolNameSet Symbols;
};

/// A symbol query that returns results via a callback when results are
///        ready.
///
/// makes a callback when all symbols are available.
class AsynchronousSymbolQuery {
public:
  /// Callback to notify client that symbols have been resolved.
  using SymbolsResolvedCallback = std::function<void(Expected<SymbolMap>)>;

  /// Callback to notify client that symbols are ready for execution.
  using SymbolsReadyCallback = std::function<void(Error)>;

  /// Create a query for the given symbols, notify-resolved and
  ///        notify-ready callbacks.
  AsynchronousSymbolQuery(const SymbolNameSet &Symbols,
                          SymbolsResolvedCallback NotifySymbolsResolved,
                          SymbolsReadyCallback NotifySymbolsReady);

  /// Notify client that the query failed.
  ///
  /// If the notify-resolved callback has not been made yet, then it is called
  /// with the given error, and the notify-finalized callback is never made.
  ///
  /// If the notify-resolved callback has already been made then then the
  /// notify-finalized callback is called with the given error.
  ///
  /// It is illegal to call setFailed after both callbacks have been made.
  void notifyMaterializationFailed(Error Err);

  /// Set the resolved symbol information for the given symbol name.
  ///
  /// If this symbol was the last one not resolved, this will trigger a call to
  /// the notify-finalized callback passing the completed sybol map.
  void resolve(SymbolStringPtr Name, JITEvaluatedSymbol Sym);

  /// Notify the query that a requested symbol is ready for execution.
  ///
  /// This decrements the query's internal count of not-yet-ready symbols. If
  /// this call to notifySymbolFinalized sets the counter to zero, it will call
  /// the notify-finalized callback with Error::success as the value.
  void finalizeSymbol();

private:
  SymbolMap Symbols;
  size_t OutstandingResolutions = 0;
  size_t OutstandingFinalizations = 0;
  SymbolsResolvedCallback NotifySymbolsResolved;
  SymbolsReadyCallback NotifySymbolsReady;
};

/// SymbolResolver is a composable interface for looking up symbol flags
///        and addresses using the AsynchronousSymbolQuery type. It will
///        eventually replace the LegacyJITSymbolResolver interface as the
///        stardard ORC symbol resolver type.
class SymbolResolver {
public:
  virtual ~SymbolResolver() = default;

  /// Returns the flags for each symbol in Symbols that can be found,
  ///        along with the set of symbol that could not be found.
  virtual SymbolNameSet lookupFlags(SymbolFlagsMap &Flags,
                                    const SymbolNameSet &Symbols) = 0;

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

/// Tracks responsibility for materialization.
///
/// An instance of this class is passed to MaterializationUnits when their
/// materialize method is called. It allows MaterializationUnits to resolve and
/// finalize symbols, or abandon materialization by notifying any unmaterialized
/// symbols of an error.
class MaterializationResponsibility {
public:
  /// Create a MaterializationResponsibility for the given VSO and
  ///        initial symbols.
  MaterializationResponsibility(VSO &V, SymbolFlagsMap SymbolFlags);

  MaterializationResponsibility(MaterializationResponsibility &&) = default;
  MaterializationResponsibility &
  operator=(MaterializationResponsibility &&) = default;

  /// Destruct a MaterializationResponsibility instance. In debug mode
  ///        this asserts that all symbols being tracked have been either
  ///        finalized or notified of an error.
  ~MaterializationResponsibility();

  /// Returns the target VSO that these symbols are being materialized
  ///        into.
  const VSO &getTargetVSO() const { return V; }

  /// Resolves the given symbols. Individual calls to this method may
  ///        resolve a subset of the symbols, but all symbols must have been
  ///        resolved prior to calling finalize.
  void resolve(const SymbolMap &Symbols);

  /// Finalizes all symbols tracked by this instance.
  void finalize();

  /// Notify all unfinalized symbols that an error has occurred.
  ///        This method should be called if materialization of any symbol is
  ///        abandoned.
  void notifyMaterializationFailed();

  /// Transfers responsibility for the given symbols to a new
  ///        MaterializationResponsibility class. This is useful if a
  ///        MaterializationUnit wants to transfer responsibility for a subset
  ///        of symbols to another MaterializationUnit or utility.
  MaterializationResponsibility delegate(SymbolNameSet Symbols);

private:
  VSO &V;
  SymbolFlagsMap SymbolFlags;
};

/// A MaterializationUnit represents a set of symbol definitions that can
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

  /// Return the set of symbols that this source provides.
  virtual SymbolFlagsMap getSymbols() = 0;

  /// Implementations of this method should materialize all symbols
  ///        in the materialzation unit, except for those that have been
  ///        previously discarded.
  virtual void materialize(MaterializationResponsibility R) = 0;

  /// Implementations of this method should discard the given symbol
  ///        from the source (e.g. if the source is an LLVM IR Module and the
  ///        symbol is a function, delete the function body or mark it available
  ///        externally).
  virtual void discard(const VSO &V, SymbolStringPtr Name) = 0;

private:
  virtual void anchor();
};

/// Represents a dynamic linkage unit in a JIT process.
///
/// VSO acts as a symbol table (symbol definitions can be set and the dylib
/// queried to find symbol addresses) and as a key for tracking resources
/// (since a VSO's address is fixed).
class VSO {
  friend class ExecutionSession;
  friend class MaterializationResponsibility;

public:
  enum RelativeLinkageStrength {
    NewDefinitionIsStronger,
    DuplicateDefinition,
    ExistingDefinitionIsStronger
  };

  using SetDefinitionsResult =
      std::map<SymbolStringPtr, RelativeLinkageStrength>;

  struct Materializer {
  public:
    Materializer(std::unique_ptr<MaterializationUnit> MU,
                 MaterializationResponsibility R);
    void operator()();

  private:
    std::unique_ptr<MaterializationUnit> MU;
    MaterializationResponsibility R;
  };

  using MaterializerList = std::vector<Materializer>;

  struct LookupResult {
    MaterializerList Materializers;
    SymbolNameSet UnresolvedSymbols;
  };

  VSO() = default;

  VSO(const VSO &) = delete;
  VSO &operator=(const VSO &) = delete;
  VSO(VSO &&) = delete;
  VSO &operator=(VSO &&) = delete;

  /// Compare new linkage with existing linkage.
  static RelativeLinkageStrength
  compareLinkage(Optional<JITSymbolFlags> OldFlags, JITSymbolFlags NewFlags);

  /// Compare new linkage with an existing symbol's linkage.
  RelativeLinkageStrength compareLinkage(SymbolStringPtr Name,
                                         JITSymbolFlags NewFlags) const;

  /// Adds the given symbols to the mapping as resolved, finalized
  ///        symbols.
  ///
  /// FIXME: We can take this by const-ref once symbol-based laziness is
  ///        removed.
  Error define(SymbolMap NewSymbols);

  /// Adds the given symbols to the mapping as lazy symbols.
  Error defineLazy(std::unique_ptr<MaterializationUnit> Source);

  /// Look up the flags for the given symbols.
  ///
  /// Returns the flags for the give symbols, together with the set of symbols
  /// not found.
  SymbolNameSet lookupFlags(SymbolFlagsMap &Flags, SymbolNameSet Symbols);

  /// Apply the given query to the given symbols in this VSO.
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
  /// Add the given symbol/address mappings to the dylib, but do not
  ///        mark the symbols as finalized yet.
  void resolve(const SymbolMap &SymbolValues);

  /// Finalize the given symbols.
  void finalize(const SymbolNameSet &SymbolsToFinalize);

  /// Notify the VSO that the given symbols failed to materialized.
  void notifyMaterializationFailed(const SymbolNameSet &Names);

  class UnmaterializedInfo {
  public:
    UnmaterializedInfo(std::unique_ptr<MaterializationUnit> MU);
    void discard(VSO &V, SymbolStringPtr Name);

    std::unique_ptr<MaterializationUnit> MU;
    SymbolFlagsMap Symbols;
  };

  using UnmaterializedInfoList = std::list<UnmaterializedInfo>;

  using UnmaterializedInfoIterator = UnmaterializedInfoList::iterator;

  class MaterializingInfo {
  public:
    using QueryList = std::vector<std::shared_ptr<AsynchronousSymbolQuery>>;

    QueryList PendingResolution;
    QueryList PendingFinalization;
  };

  using MaterializingInfoMap = std::map<SymbolStringPtr, MaterializingInfo>;

  using MaterializingInfoIterator = MaterializingInfoMap::iterator;

  class SymbolTableEntry {
  public:
    SymbolTableEntry(JITSymbolFlags SymbolFlags,
                     UnmaterializedInfoIterator UnmaterializedInfoItr);
    SymbolTableEntry(JITSymbolFlags SymbolFlags);
    SymbolTableEntry(JITEvaluatedSymbol Sym);
    SymbolTableEntry(SymbolTableEntry &&Other);
    SymbolTableEntry &operator=(SymbolTableEntry &&Other);
    ~SymbolTableEntry();

    // Change definition due to override. Only usable prior to materialization.
    void replaceWith(VSO &V, SymbolStringPtr Name, JITEvaluatedSymbol Sym);

    // Change definition due to override. Only usable prior to materialization.
    void replaceWith(VSO &V, SymbolStringPtr Name, JITSymbolFlags Flags,
                     UnmaterializedInfoIterator NewUMII);

    // Abandon old definition and move to materializing state.
    // There is no need to call notifyMaterializing after this.
    void replaceMaterializing(VSO &V, SymbolStringPtr Name,
                              JITSymbolFlags NewFlags);

    // Notify this entry that it is being materialized.
    void notifyMaterializing();

    // Move entry to resolved state.
    void resolve(VSO &V, JITEvaluatedSymbol Sym);

    // Move entry to finalized state.
    void finalize();

    JITSymbolFlags Flags;

    union {
      JITTargetAddress Address;
      UnmaterializedInfoIterator UMII;
    };

  private:
    void destroy();
  };

  std::map<SymbolStringPtr, SymbolTableEntry> Symbols;
  UnmaterializedInfoList UnmaterializedInfos;
  MaterializingInfoMap MaterializingInfos;
};

/// An ExecutionSession represents a running JIT program.
class ExecutionSession {
public:
  using ErrorReporter = std::function<void(Error)>;

  /// Construct an ExecutionEngine.
  ///
  /// SymbolStringPools may be shared between ExecutionSessions.
  ExecutionSession(std::shared_ptr<SymbolStringPool> SSP = nullptr)
    : SSP(SSP ? std::move(SSP) : std::make_shared<SymbolStringPool>()) {}

  /// Returns the SymbolStringPool for this ExecutionSession.
  SymbolStringPool &getSymbolStringPool() const { return *SSP; }

  /// Set the error reporter function.
  void setErrorReporter(ErrorReporter ReportError) {
    this->ReportError = std::move(ReportError);
  }

  /// Report a error for this execution session.
  ///
  /// Unhandled errors can be sent here to log them.
  void reportError(Error Err) { ReportError(std::move(Err)); }

  /// Allocate a module key for a new module to add to the JIT.
  VModuleKey allocateVModule() { return ++LastKey; }

  /// Return a module key to the ExecutionSession so that it can be
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
  void operator()(VSO::Materializer M) { M(); }
};

/// Materialization function object wrapper for the lookup method.
using MaterializationDispatcher = std::function<void(VSO::Materializer M)>;

/// Look up a set of symbols by searching a list of VSOs.
///
/// All VSOs in the list should be non-null.
Expected<SymbolMap> lookup(const std::vector<VSO *> &VSOs, SymbolNameSet Names,
                           MaterializationDispatcher DispatchMaterialization);

/// Look up a symbol by searching a list of VSOs.
Expected<JITEvaluatedSymbol>
lookup(const std::vector<VSO *> VSOs, SymbolStringPtr Name,
       MaterializationDispatcher DispatchMaterialization);

} // End namespace orc
} // End namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_CORE_H
