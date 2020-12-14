//===------ Core.h -- Core ORC APIs (Layer, JITDylib, etc.) -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains core ORC APIs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_CORE_H
#define LLVM_EXECUTIONENGINE_ORC_CORE_H

#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/SymbolStringPool.h"
#include "llvm/ExecutionEngine/OrcV1Deprecation.h"
#include "llvm/Support/Debug.h"

#include <atomic>
#include <memory>
#include <vector>

namespace llvm {
namespace orc {

// Forward declare some classes.
class AsynchronousSymbolQuery;
class ExecutionSession;
class MaterializationUnit;
class MaterializationResponsibility;
class JITDylib;
class ResourceTracker;
class InProgressLookupState;

enum class SymbolState : uint8_t;

using ResourceTrackerSP = IntrusiveRefCntPtr<ResourceTracker>;
using JITDylibSP = IntrusiveRefCntPtr<JITDylib>;

using ResourceKey = uintptr_t;

/// API to remove / transfer ownership of JIT resources.
class ResourceTracker : public ThreadSafeRefCountedBase<ResourceTracker> {
private:
  friend class ExecutionSession;
  friend class JITDylib;
  friend class MaterializationResponsibility;

public:
  ResourceTracker(const ResourceTracker &) = delete;
  ResourceTracker &operator=(const ResourceTracker &) = delete;
  ResourceTracker(ResourceTracker &&) = delete;
  ResourceTracker &operator=(ResourceTracker &&) = delete;

  ~ResourceTracker();

  /// Return the JITDylib targeted by this tracker.
  JITDylib &getJITDylib() const {
    return *reinterpret_cast<JITDylib *>(JDAndFlag.load() &
                                         ~static_cast<uintptr_t>(1));
  }

  /// Remove all resources associated with this key.
  Error remove();

  /// Transfer all resources associated with this key to the given
  /// tracker, which must target the same JITDylib as this one.
  void transferTo(ResourceTracker &DstRT);

  /// Return true if this tracker has become defunct.
  bool isDefunct() const { return JDAndFlag.load() & 0x1; }

  /// Returns the key associated with this tracker.
  /// This method should not be used except for debug logging: there is no
  /// guarantee that the returned value will remain valid.
  ResourceKey getKeyUnsafe() const { return reinterpret_cast<uintptr_t>(this); }

private:
  ResourceTracker(JITDylibSP JD);

  void makeDefunct();

  std::atomic_uintptr_t JDAndFlag;
};

/// Listens for ResourceTracker operations.
class ResourceManager {
public:
  virtual ~ResourceManager();
  virtual Error handleRemoveResources(ResourceKey K) = 0;
  virtual void handleTransferResources(ResourceKey DstK, ResourceKey SrcK) = 0;
};

/// A set of symbol names (represented by SymbolStringPtrs for
//         efficiency).
using SymbolNameSet = DenseSet<SymbolStringPtr>;

/// A vector of symbol names.
using SymbolNameVector = std::vector<SymbolStringPtr>;

/// A map from symbol names (as SymbolStringPtrs) to JITSymbols
/// (address/flags pairs).
using SymbolMap = DenseMap<SymbolStringPtr, JITEvaluatedSymbol>;

/// A map from symbol names (as SymbolStringPtrs) to JITSymbolFlags.
using SymbolFlagsMap = DenseMap<SymbolStringPtr, JITSymbolFlags>;

/// A map from JITDylibs to sets of symbols.
using SymbolDependenceMap = DenseMap<JITDylib *, SymbolNameSet>;

/// Lookup flags that apply to each dylib in the search order for a lookup.
///
/// If MatchHiddenSymbolsOnly is used (the default) for a given dylib, then
/// only symbols in that Dylib's interface will be searched. If
/// MatchHiddenSymbols is used then symbols with hidden visibility will match
/// as well.
enum class JITDylibLookupFlags { MatchExportedSymbolsOnly, MatchAllSymbols };

/// Lookup flags that apply to each symbol in a lookup.
///
/// If RequiredSymbol is used (the default) for a given symbol then that symbol
/// must be found during the lookup or the lookup will fail returning a
/// SymbolNotFound error. If WeaklyReferencedSymbol is used and the given
/// symbol is not found then the query will continue, and no result for the
/// missing symbol will be present in the result (assuming the rest of the
/// lookup succeeds).
enum class SymbolLookupFlags { RequiredSymbol, WeaklyReferencedSymbol };

/// Describes the kind of lookup being performed. The lookup kind is passed to
/// symbol generators (if they're invoked) to help them determine what
/// definitions to generate.
///
/// Static -- Lookup is being performed as-if at static link time (e.g.
///           generators representing static archives should pull in new
///           definitions).
///
/// DLSym -- Lookup is being performed as-if at runtime (e.g. generators
///          representing static archives should not pull in new definitions).
enum class LookupKind { Static, DLSym };

/// A list of (JITDylib*, JITDylibLookupFlags) pairs to be used as a search
/// order during symbol lookup.
using JITDylibSearchOrder =
    std::vector<std::pair<JITDylib *, JITDylibLookupFlags>>;

/// Convenience function for creating a search order from an ArrayRef of
/// JITDylib*, all with the same flags.
inline JITDylibSearchOrder makeJITDylibSearchOrder(
    ArrayRef<JITDylib *> JDs,
    JITDylibLookupFlags Flags = JITDylibLookupFlags::MatchExportedSymbolsOnly) {
  JITDylibSearchOrder O;
  O.reserve(JDs.size());
  for (auto *JD : JDs)
    O.push_back(std::make_pair(JD, Flags));
  return O;
}

/// A set of symbols to look up, each associated with a SymbolLookupFlags
/// value.
///
/// This class is backed by a vector and optimized for fast insertion,
/// deletion and iteration. It does not guarantee a stable order between
/// operations, and will not automatically detect duplicate elements (they
/// can be manually checked by calling the validate method).
class SymbolLookupSet {
public:
  using value_type = std::pair<SymbolStringPtr, SymbolLookupFlags>;
  using UnderlyingVector = std::vector<value_type>;
  using iterator = UnderlyingVector::iterator;
  using const_iterator = UnderlyingVector::const_iterator;

  SymbolLookupSet() = default;

  explicit SymbolLookupSet(
      SymbolStringPtr Name,
      SymbolLookupFlags Flags = SymbolLookupFlags::RequiredSymbol) {
    add(std::move(Name), Flags);
  }

  /// Construct a SymbolLookupSet from an initializer list of SymbolStringPtrs.
  explicit SymbolLookupSet(
      std::initializer_list<SymbolStringPtr> Names,
      SymbolLookupFlags Flags = SymbolLookupFlags::RequiredSymbol) {
    Symbols.reserve(Names.size());
    for (auto &Name : Names)
      add(std::move(Name), Flags);
  }

  /// Construct a SymbolLookupSet from a SymbolNameSet with the given
  /// Flags used for each value.
  explicit SymbolLookupSet(
      const SymbolNameSet &Names,
      SymbolLookupFlags Flags = SymbolLookupFlags::RequiredSymbol) {
    Symbols.reserve(Names.size());
    for (const auto &Name : Names)
      add(Name, Flags);
  }

  /// Construct a SymbolLookupSet from a vector of symbols with the given Flags
  /// used for each value.
  /// If the ArrayRef contains duplicates it is up to the client to remove these
  /// before using this instance for lookup.
  explicit SymbolLookupSet(
      ArrayRef<SymbolStringPtr> Names,
      SymbolLookupFlags Flags = SymbolLookupFlags::RequiredSymbol) {
    Symbols.reserve(Names.size());
    for (const auto &Name : Names)
      add(Name, Flags);
  }

  /// Add an element to the set. The client is responsible for checking that
  /// duplicates are not added.
  SymbolLookupSet &
  add(SymbolStringPtr Name,
      SymbolLookupFlags Flags = SymbolLookupFlags::RequiredSymbol) {
    Symbols.push_back(std::make_pair(std::move(Name), Flags));
    return *this;
  }

  /// Quickly append one lookup set to another.
  SymbolLookupSet &append(SymbolLookupSet Other) {
    Symbols.reserve(Symbols.size() + Other.size());
    for (auto &KV : Other)
      Symbols.push_back(std::move(KV));
    return *this;
  }

  bool empty() const { return Symbols.empty(); }
  UnderlyingVector::size_type size() const { return Symbols.size(); }
  iterator begin() { return Symbols.begin(); }
  iterator end() { return Symbols.end(); }
  const_iterator begin() const { return Symbols.begin(); }
  const_iterator end() const { return Symbols.end(); }

  /// Removes the Ith element of the vector, replacing it with the last element.
  void remove(UnderlyingVector::size_type I) {
    std::swap(Symbols[I], Symbols.back());
    Symbols.pop_back();
  }

  /// Removes the element pointed to by the given iterator. This iterator and
  /// all subsequent ones (including end()) are invalidated.
  void remove(iterator I) { remove(I - begin()); }

  /// Removes all elements matching the given predicate, which must be callable
  /// as bool(const SymbolStringPtr &, SymbolLookupFlags Flags).
  template <typename PredFn> void remove_if(PredFn &&Pred) {
    UnderlyingVector::size_type I = 0;
    while (I != Symbols.size()) {
      const auto &Name = Symbols[I].first;
      auto Flags = Symbols[I].second;
      if (Pred(Name, Flags))
        remove(I);
      else
        ++I;
    }
  }

  /// Loop over the elements of this SymbolLookupSet, applying the Body function
  /// to each one. Body must be callable as
  /// bool(const SymbolStringPtr &, SymbolLookupFlags).
  /// If Body returns true then the element just passed in is removed from the
  /// set. If Body returns false then the element is retained.
  template <typename BodyFn>
  auto forEachWithRemoval(BodyFn &&Body) -> std::enable_if_t<
      std::is_same<decltype(Body(std::declval<const SymbolStringPtr &>(),
                                 std::declval<SymbolLookupFlags>())),
                   bool>::value> {
    UnderlyingVector::size_type I = 0;
    while (I != Symbols.size()) {
      const auto &Name = Symbols[I].first;
      auto Flags = Symbols[I].second;
      if (Body(Name, Flags))
        remove(I);
      else
        ++I;
    }
  }

  /// Loop over the elements of this SymbolLookupSet, applying the Body function
  /// to each one. Body must be callable as
  /// Expected<bool>(const SymbolStringPtr &, SymbolLookupFlags).
  /// If Body returns a failure value, the loop exits immediately. If Body
  /// returns true then the element just passed in is removed from the set. If
  /// Body returns false then the element is retained.
  template <typename BodyFn>
  auto forEachWithRemoval(BodyFn &&Body) -> std::enable_if_t<
      std::is_same<decltype(Body(std::declval<const SymbolStringPtr &>(),
                                 std::declval<SymbolLookupFlags>())),
                   Expected<bool>>::value,
      Error> {
    UnderlyingVector::size_type I = 0;
    while (I != Symbols.size()) {
      const auto &Name = Symbols[I].first;
      auto Flags = Symbols[I].second;
      auto Remove = Body(Name, Flags);
      if (!Remove)
        return Remove.takeError();
      if (*Remove)
        remove(I);
      else
        ++I;
    }
    return Error::success();
  }

  /// Construct a SymbolNameVector from this instance by dropping the Flags
  /// values.
  SymbolNameVector getSymbolNames() const {
    SymbolNameVector Names;
    Names.reserve(Symbols.size());
    for (auto &KV : Symbols)
      Names.push_back(KV.first);
    return Names;
  }

  /// Sort the lookup set by pointer value. This sort is fast but sensitive to
  /// allocation order and so should not be used where a consistent order is
  /// required.
  void sortByAddress() {
    llvm::sort(Symbols, [](const value_type &LHS, const value_type &RHS) {
      return LHS.first < RHS.first;
    });
  }

  /// Sort the lookup set lexicographically. This sort is slow but the order
  /// is unaffected by allocation order.
  void sortByName() {
    llvm::sort(Symbols, [](const value_type &LHS, const value_type &RHS) {
      return *LHS.first < *RHS.first;
    });
  }

  /// Remove any duplicate elements. If a SymbolLookupSet is not duplicate-free
  /// by construction, this method can be used to turn it into a proper set.
  void removeDuplicates() {
    sortByAddress();
    auto LastI = std::unique(Symbols.begin(), Symbols.end());
    Symbols.erase(LastI, Symbols.end());
  }

#ifndef NDEBUG
  /// Returns true if this set contains any duplicates. This should only be used
  /// in assertions.
  bool containsDuplicates() {
    if (Symbols.size() < 2)
      return false;
    sortByAddress();
    for (UnderlyingVector::size_type I = 1; I != Symbols.size(); ++I)
      if (Symbols[I].first == Symbols[I - 1].first)
        return true;
    return false;
  }
#endif

private:
  UnderlyingVector Symbols;
};

struct SymbolAliasMapEntry {
  SymbolAliasMapEntry() = default;
  SymbolAliasMapEntry(SymbolStringPtr Aliasee, JITSymbolFlags AliasFlags)
      : Aliasee(std::move(Aliasee)), AliasFlags(AliasFlags) {}

  SymbolStringPtr Aliasee;
  JITSymbolFlags AliasFlags;
};

/// A map of Symbols to (Symbol, Flags) pairs.
using SymbolAliasMap = DenseMap<SymbolStringPtr, SymbolAliasMapEntry>;

/// Callback to notify client that symbols have been resolved.
using SymbolsResolvedCallback = unique_function<void(Expected<SymbolMap>)>;

/// Callback to register the dependencies for a given query.
using RegisterDependenciesFunction =
    std::function<void(const SymbolDependenceMap &)>;

/// This can be used as the value for a RegisterDependenciesFunction if there
/// are no dependants to register with.
extern RegisterDependenciesFunction NoDependenciesToRegister;

class ResourceTrackerDefunct : public ErrorInfo<ResourceTrackerDefunct> {
public:
  static char ID;

  ResourceTrackerDefunct(ResourceTrackerSP RT);
  std::error_code convertToErrorCode() const override;
  void log(raw_ostream &OS) const override;

private:
  ResourceTrackerSP RT;
};

/// Used to notify a JITDylib that the given set of symbols failed to
/// materialize.
class FailedToMaterialize : public ErrorInfo<FailedToMaterialize> {
public:
  static char ID;

  FailedToMaterialize(std::shared_ptr<SymbolDependenceMap> Symbols);
  std::error_code convertToErrorCode() const override;
  void log(raw_ostream &OS) const override;
  const SymbolDependenceMap &getSymbols() const { return *Symbols; }

private:
  std::shared_ptr<SymbolDependenceMap> Symbols;
};

/// Used to notify clients when symbols can not be found during a lookup.
class SymbolsNotFound : public ErrorInfo<SymbolsNotFound> {
public:
  static char ID;

  SymbolsNotFound(SymbolNameSet Symbols);
  SymbolsNotFound(SymbolNameVector Symbols);
  std::error_code convertToErrorCode() const override;
  void log(raw_ostream &OS) const override;
  const SymbolNameVector &getSymbols() const { return Symbols; }

private:
  SymbolNameVector Symbols;
};

/// Used to notify clients that a set of symbols could not be removed.
class SymbolsCouldNotBeRemoved : public ErrorInfo<SymbolsCouldNotBeRemoved> {
public:
  static char ID;

  SymbolsCouldNotBeRemoved(SymbolNameSet Symbols);
  std::error_code convertToErrorCode() const override;
  void log(raw_ostream &OS) const override;
  const SymbolNameSet &getSymbols() const { return Symbols; }

private:
  SymbolNameSet Symbols;
};

/// Errors of this type should be returned if a module fails to include
/// definitions that are claimed by the module's associated
/// MaterializationResponsibility. If this error is returned it is indicative of
/// a broken transformation / compiler / object cache.
class MissingSymbolDefinitions : public ErrorInfo<MissingSymbolDefinitions> {
public:
  static char ID;

  MissingSymbolDefinitions(std::string ModuleName, SymbolNameVector Symbols)
    : ModuleName(std::move(ModuleName)), Symbols(std::move(Symbols)) {}
  std::error_code convertToErrorCode() const override;
  void log(raw_ostream &OS) const override;
  const std::string &getModuleName() const { return ModuleName; }
  const SymbolNameVector &getSymbols() const { return Symbols; }
private:
  std::string ModuleName;
  SymbolNameVector Symbols;
};

/// Errors of this type should be returned if a module contains definitions for
/// symbols that are not claimed by the module's associated
/// MaterializationResponsibility. If this error is returned it is indicative of
/// a broken transformation / compiler / object cache.
class UnexpectedSymbolDefinitions : public ErrorInfo<UnexpectedSymbolDefinitions> {
public:
  static char ID;

  UnexpectedSymbolDefinitions(std::string ModuleName, SymbolNameVector Symbols)
    : ModuleName(std::move(ModuleName)), Symbols(std::move(Symbols)) {}
  std::error_code convertToErrorCode() const override;
  void log(raw_ostream &OS) const override;
  const std::string &getModuleName() const { return ModuleName; }
  const SymbolNameVector &getSymbols() const { return Symbols; }
private:
  std::string ModuleName;
  SymbolNameVector Symbols;
};

/// Tracks responsibility for materialization, and mediates interactions between
/// MaterializationUnits and JDs.
///
/// An instance of this class is passed to MaterializationUnits when their
/// materialize method is called. It allows MaterializationUnits to resolve and
/// emit symbols, or abandon materialization by notifying any unmaterialized
/// symbols of an error.
class MaterializationResponsibility {
  friend class ExecutionSession;

public:
  MaterializationResponsibility(MaterializationResponsibility &&) = delete;
  MaterializationResponsibility &
  operator=(MaterializationResponsibility &&) = delete;

  /// Destruct a MaterializationResponsibility instance. In debug mode
  ///        this asserts that all symbols being tracked have been either
  ///        emitted or notified of an error.
  ~MaterializationResponsibility();

  /// Returns the ResourceTracker for this instance.
  template <typename Func> Error withResourceKeyDo(Func &&F) const;

  /// Returns the target JITDylib that these symbols are being materialized
  ///        into.
  JITDylib &getTargetJITDylib() const { return *JD; }

  /// Returns the ExecutionSession for this instance.
  ExecutionSession &getExecutionSession();

  /// Returns the symbol flags map for this responsibility instance.
  /// Note: The returned flags may have transient flags (Lazy, Materializing)
  /// set. These should be stripped with JITSymbolFlags::stripTransientFlags
  /// before using.
  const SymbolFlagsMap &getSymbols() const { return SymbolFlags; }

  /// Returns the initialization pseudo-symbol, if any. This symbol will also
  /// be present in the SymbolFlagsMap for this MaterializationResponsibility
  /// object.
  const SymbolStringPtr &getInitializerSymbol() const { return InitSymbol; }

  /// Returns the names of any symbols covered by this
  /// MaterializationResponsibility object that have queries pending. This
  /// information can be used to return responsibility for unrequested symbols
  /// back to the JITDylib via the delegate method.
  SymbolNameSet getRequestedSymbols() const;

  /// Notifies the target JITDylib that the given symbols have been resolved.
  /// This will update the given symbols' addresses in the JITDylib, and notify
  /// any pending queries on the given symbols of their resolution. The given
  /// symbols must be ones covered by this MaterializationResponsibility
  /// instance. Individual calls to this method may resolve a subset of the
  /// symbols, but all symbols must have been resolved prior to calling emit.
  ///
  /// This method will return an error if any symbols being resolved have been
  /// moved to the error state due to the failure of a dependency. If this
  /// method returns an error then clients should log it and call
  /// failMaterialize. If no dependencies have been registered for the
  /// symbols covered by this MaterializationResponsibiility then this method
  /// is guaranteed to return Error::success() and can be wrapped with cantFail.
  Error notifyResolved(const SymbolMap &Symbols);

  /// Notifies the target JITDylib (and any pending queries on that JITDylib)
  /// that all symbols covered by this MaterializationResponsibility instance
  /// have been emitted.
  ///
  /// This method will return an error if any symbols being resolved have been
  /// moved to the error state due to the failure of a dependency. If this
  /// method returns an error then clients should log it and call
  /// failMaterialize. If no dependencies have been registered for the
  /// symbols covered by this MaterializationResponsibiility then this method
  /// is guaranteed to return Error::success() and can be wrapped with cantFail.
  Error notifyEmitted();

  /// Attempt to claim responsibility for new definitions. This method can be
  /// used to claim responsibility for symbols that are added to a
  /// materialization unit during the compilation process (e.g. literal pool
  /// symbols). Symbol linkage rules are the same as for symbols that are
  /// defined up front: duplicate strong definitions will result in errors.
  /// Duplicate weak definitions will be discarded (in which case they will
  /// not be added to this responsibility instance).
  ///
  ///   This method can be used by materialization units that want to add
  /// additional symbols at materialization time (e.g. stubs, compile
  /// callbacks, metadata).
  Error defineMaterializing(SymbolFlagsMap SymbolFlags);

  /// Define the given symbols as non-existent, removing it from the symbol
  /// table and notifying any pending queries. Queries that lookup up the
  /// symbol using the SymbolLookupFlags::WeaklyReferencedSymbol flag will
  /// behave as if the symbol had not been matched in the first place. Queries
  /// that required this symbol will fail with a missing symbol definition
  /// error.
  ///
  /// This method is intended to support cleanup of special symbols like
  /// initializer symbols: Queries using
  /// SymbolLookupFlags::WeaklyReferencedSymbol can be used to trigger their
  /// emission, and this method can be used to remove them from the JITDylib
  /// once materialization is complete.
  void defineNonExistent(ArrayRef<SymbolStringPtr> Symbols);

  /// Notify all not-yet-emitted covered by this MaterializationResponsibility
  /// instance that an error has occurred.
  /// This will remove all symbols covered by this MaterializationResponsibilty
  /// from the target JITDylib, and send an error to any queries waiting on
  /// these symbols.
  void failMaterialization();

  /// Transfers responsibility to the given MaterializationUnit for all
  /// symbols defined by that MaterializationUnit. This allows
  /// materializers to break up work based on run-time information (e.g.
  /// by introspecting which symbols have actually been looked up and
  /// materializing only those).
  Error replace(std::unique_ptr<MaterializationUnit> MU);

  /// Delegates responsibility for the given symbols to the returned
  /// materialization responsibility. Useful for breaking up work between
  /// threads, or different kinds of materialization processes.
  Expected<std::unique_ptr<MaterializationResponsibility>>
  delegate(const SymbolNameSet &Symbols);

  void addDependencies(const SymbolStringPtr &Name,
                       const SymbolDependenceMap &Dependencies);

  /// Add dependencies that apply to all symbols covered by this instance.
  void addDependenciesForAll(const SymbolDependenceMap &Dependencies);

private:
  /// Create a MaterializationResponsibility for the given JITDylib and
  ///        initial symbols.
  MaterializationResponsibility(JITDylibSP JD, SymbolFlagsMap SymbolFlags,
                                SymbolStringPtr InitSymbol)
      : JD(std::move(JD)), SymbolFlags(std::move(SymbolFlags)),
        InitSymbol(std::move(InitSymbol)) {
    assert(this->JD && "Cannot initialize with null JITDylib");
    assert(!this->SymbolFlags.empty() && "Materializing nothing?");
  }

  JITDylibSP JD;
  SymbolFlagsMap SymbolFlags;
  SymbolStringPtr InitSymbol;
};

/// A MaterializationUnit represents a set of symbol definitions that can
///        be materialized as a group, or individually discarded (when
///        overriding definitions are encountered).
///
/// MaterializationUnits are used when providing lazy definitions of symbols to
/// JITDylibs. The JITDylib will call materialize when the address of a symbol
/// is requested via the lookup method. The JITDylib will call discard if a
/// stronger definition is added or already present.
class MaterializationUnit {
  friend class ExecutionSession;
  friend class JITDylib;

public:
  MaterializationUnit(SymbolFlagsMap InitalSymbolFlags,
                      SymbolStringPtr InitSymbol)
      : SymbolFlags(std::move(InitalSymbolFlags)),
        InitSymbol(std::move(InitSymbol)) {
    assert((!this->InitSymbol || this->SymbolFlags.count(this->InitSymbol)) &&
           "If set, InitSymbol should appear in InitialSymbolFlags map");
  }

  virtual ~MaterializationUnit() {}

  /// Return the name of this materialization unit. Useful for debugging
  /// output.
  virtual StringRef getName() const = 0;

  /// Return the set of symbols that this source provides.
  const SymbolFlagsMap &getSymbols() const { return SymbolFlags; }

  /// Returns the initialization symbol for this MaterializationUnit (if any).
  const SymbolStringPtr &getInitializerSymbol() const { return InitSymbol; }

  /// Implementations of this method should materialize all symbols
  ///        in the materialzation unit, except for those that have been
  ///        previously discarded.
  virtual void
  materialize(std::unique_ptr<MaterializationResponsibility> R) = 0;

  /// Called by JITDylibs to notify MaterializationUnits that the given symbol
  /// has been overridden.
  void doDiscard(const JITDylib &JD, const SymbolStringPtr &Name) {
    SymbolFlags.erase(Name);
    discard(JD, std::move(Name));
  }

protected:
  SymbolFlagsMap SymbolFlags;
  SymbolStringPtr InitSymbol;

private:
  virtual void anchor();

  /// Implementations of this method should discard the given symbol
  ///        from the source (e.g. if the source is an LLVM IR Module and the
  ///        symbol is a function, delete the function body or mark it available
  ///        externally).
  virtual void discard(const JITDylib &JD, const SymbolStringPtr &Name) = 0;
};

/// A MaterializationUnit implementation for pre-existing absolute symbols.
///
/// All symbols will be resolved and marked ready as soon as the unit is
/// materialized.
class AbsoluteSymbolsMaterializationUnit : public MaterializationUnit {
public:
  AbsoluteSymbolsMaterializationUnit(SymbolMap Symbols);

  StringRef getName() const override;

private:
  void materialize(std::unique_ptr<MaterializationResponsibility> R) override;
  void discard(const JITDylib &JD, const SymbolStringPtr &Name) override;
  static SymbolFlagsMap extractFlags(const SymbolMap &Symbols);

  SymbolMap Symbols;
};

/// Create an AbsoluteSymbolsMaterializationUnit with the given symbols.
/// Useful for inserting absolute symbols into a JITDylib. E.g.:
/// \code{.cpp}
///   JITDylib &JD = ...;
///   SymbolStringPtr Foo = ...;
///   JITEvaluatedSymbol FooSym = ...;
///   if (auto Err = JD.define(absoluteSymbols({{Foo, FooSym}})))
///     return Err;
/// \endcode
///
inline std::unique_ptr<AbsoluteSymbolsMaterializationUnit>
absoluteSymbols(SymbolMap Symbols) {
  return std::make_unique<AbsoluteSymbolsMaterializationUnit>(
      std::move(Symbols));
}

/// A materialization unit for symbol aliases. Allows existing symbols to be
/// aliased with alternate flags.
class ReExportsMaterializationUnit : public MaterializationUnit {
public:
  /// SourceJD is allowed to be nullptr, in which case the source JITDylib is
  /// taken to be whatever JITDylib these definitions are materialized in (and
  /// MatchNonExported has no effect). This is useful for defining aliases
  /// within a JITDylib.
  ///
  /// Note: Care must be taken that no sets of aliases form a cycle, as such
  ///       a cycle will result in a deadlock when any symbol in the cycle is
  ///       resolved.
  ReExportsMaterializationUnit(JITDylib *SourceJD,
                               JITDylibLookupFlags SourceJDLookupFlags,
                               SymbolAliasMap Aliases);

  StringRef getName() const override;

private:
  void materialize(std::unique_ptr<MaterializationResponsibility> R) override;
  void discard(const JITDylib &JD, const SymbolStringPtr &Name) override;
  static SymbolFlagsMap extractFlags(const SymbolAliasMap &Aliases);

  JITDylib *SourceJD = nullptr;
  JITDylibLookupFlags SourceJDLookupFlags;
  SymbolAliasMap Aliases;
};

/// Create a ReExportsMaterializationUnit with the given aliases.
/// Useful for defining symbol aliases.: E.g., given a JITDylib JD containing
/// symbols "foo" and "bar", we can define aliases "baz" (for "foo") and "qux"
/// (for "bar") with: \code{.cpp}
///   SymbolStringPtr Baz = ...;
///   SymbolStringPtr Qux = ...;
///   if (auto Err = JD.define(symbolAliases({
///       {Baz, { Foo, JITSymbolFlags::Exported }},
///       {Qux, { Bar, JITSymbolFlags::Weak }}}))
///     return Err;
/// \endcode
inline std::unique_ptr<ReExportsMaterializationUnit>
symbolAliases(SymbolAliasMap Aliases) {
  return std::make_unique<ReExportsMaterializationUnit>(
      nullptr, JITDylibLookupFlags::MatchAllSymbols, std::move(Aliases));
}

/// Create a materialization unit for re-exporting symbols from another JITDylib
/// with alternative names/flags.
/// SourceJD will be searched using the given JITDylibLookupFlags.
inline std::unique_ptr<ReExportsMaterializationUnit>
reexports(JITDylib &SourceJD, SymbolAliasMap Aliases,
          JITDylibLookupFlags SourceJDLookupFlags =
              JITDylibLookupFlags::MatchExportedSymbolsOnly) {
  return std::make_unique<ReExportsMaterializationUnit>(
      &SourceJD, SourceJDLookupFlags, std::move(Aliases));
}

/// Build a SymbolAliasMap for the common case where you want to re-export
/// symbols from another JITDylib with the same linkage/flags.
Expected<SymbolAliasMap>
buildSimpleReexportsAAliasMap(JITDylib &SourceJD, const SymbolNameSet &Symbols);

/// Represents the state that a symbol has reached during materialization.
enum class SymbolState : uint8_t {
  Invalid,       /// No symbol should be in this state.
  NeverSearched, /// Added to the symbol table, never queried.
  Materializing, /// Queried, materialization begun.
  Resolved,      /// Assigned address, still materializing.
  Emitted,       /// Emitted to memory, but waiting on transitive dependencies.
  Ready = 0x3f   /// Ready and safe for clients to access.
};

/// A symbol query that returns results via a callback when results are
///        ready.
///
/// makes a callback when all symbols are available.
class AsynchronousSymbolQuery {
  friend class ExecutionSession;
  friend class InProgressFullLookupState;
  friend class JITDylib;
  friend class JITSymbolResolverAdapter;
  friend class MaterializationResponsibility;

public:
  /// Create a query for the given symbols. The NotifyComplete
  /// callback will be called once all queried symbols reach the given
  /// minimum state.
  AsynchronousSymbolQuery(const SymbolLookupSet &Symbols,
                          SymbolState RequiredState,
                          SymbolsResolvedCallback NotifyComplete);

  /// Notify the query that a requested symbol has reached the required state.
  void notifySymbolMetRequiredState(const SymbolStringPtr &Name,
                                    JITEvaluatedSymbol Sym);

  /// Returns true if all symbols covered by this query have been
  ///        resolved.
  bool isComplete() const { return OutstandingSymbolsCount == 0; }

  /// Call the NotifyComplete callback.
  ///
  /// This should only be called if all symbols covered by the query have
  /// reached the specified state.
  void handleComplete();

private:
  SymbolState getRequiredState() { return RequiredState; }

  void addQueryDependence(JITDylib &JD, SymbolStringPtr Name);

  void removeQueryDependence(JITDylib &JD, const SymbolStringPtr &Name);

  void dropSymbol(const SymbolStringPtr &Name);

  void handleFailed(Error Err);

  void detach();

  SymbolsResolvedCallback NotifyComplete;
  SymbolDependenceMap QueryRegistrations;
  SymbolMap ResolvedSymbols;
  size_t OutstandingSymbolsCount;
  SymbolState RequiredState;
};

/// Wraps state for a lookup-in-progress.
/// DefinitionGenerators can optionally take ownership of a LookupState object
/// to suspend a lookup-in-progress while they search for definitions.
class LookupState {
  friend class OrcV2CAPIHelper;
  friend class ExecutionSession;

public:
  ~LookupState();

  /// Continue the lookup. This can be called by DefinitionGenerators
  /// to re-start a captured query-application operation.
  void continueLookup(Error Err);

private:
  LookupState(std::unique_ptr<InProgressLookupState> IPLS);

  // For C API.
  void reset(InProgressLookupState *IPLS);

  std::unique_ptr<InProgressLookupState> IPLS;
};

/// Definition generators can be attached to JITDylibs to generate new
/// definitions for otherwise unresolved symbols during lookup.
class DefinitionGenerator {
public:
  virtual ~DefinitionGenerator();

  /// DefinitionGenerators should override this method to insert new
  /// definitions into the parent JITDylib. K specifies the kind of this
  /// lookup. JD specifies the target JITDylib being searched, and
  /// JDLookupFlags specifies whether the search should match against
  /// hidden symbols. Finally, Symbols describes the set of unresolved
  /// symbols and their associated lookup flags.
  virtual Error tryToGenerate(LookupState &LS, LookupKind K, JITDylib &JD,
                              JITDylibLookupFlags JDLookupFlags,
                              const SymbolLookupSet &LookupSet) = 0;
};

/// A symbol table that supports asynchoronous symbol queries.
///
/// Represents a virtual shared object. Instances can not be copied or moved, so
/// their addresses may be used as keys for resource management.
/// JITDylib state changes must be made via an ExecutionSession to guarantee
/// that they are synchronized with respect to other JITDylib operations.
class JITDylib : public ThreadSafeRefCountedBase<JITDylib> {
  friend class AsynchronousSymbolQuery;
  friend class ExecutionSession;
  friend class Platform;
  friend class MaterializationResponsibility;
public:

  using AsynchronousSymbolQuerySet =
    std::set<std::shared_ptr<AsynchronousSymbolQuery>>;

  JITDylib(const JITDylib &) = delete;
  JITDylib &operator=(const JITDylib &) = delete;
  JITDylib(JITDylib &&) = delete;
  JITDylib &operator=(JITDylib &&) = delete;

  /// Get the name for this JITDylib.
  const std::string &getName() const { return JITDylibName; }

  /// Get a reference to the ExecutionSession for this JITDylib.
  ExecutionSession &getExecutionSession() const { return ES; }

  /// Calls remove on all trackers currently associated with this JITDylib.
  /// Does not run static deinits.
  ///
  /// Note that removal happens outside the session lock, so new code may be
  /// added concurrently while the clear is underway, and the newly added
  /// code will *not* be cleared. Adding new code concurrently with a clear
  /// is usually a bug and should be avoided.
  Error clear();

  /// Get the default resource tracker for this JITDylib.
  ResourceTrackerSP getDefaultResourceTracker();

  /// Create a resource tracker for this JITDylib.
  ResourceTrackerSP createResourceTracker();

  /// Adds a definition generator to this JITDylib and returns a referenece to
  /// it.
  ///
  /// When JITDylibs are searched during lookup, if no existing definition of
  /// a symbol is found, then any generators that have been added are run (in
  /// the order that they were added) to potentially generate a definition.
  template <typename GeneratorT>
  GeneratorT &addGenerator(std::unique_ptr<GeneratorT> DefGenerator);

  /// Remove a definition generator from this JITDylib.
  ///
  /// The given generator must exist in this JITDylib's generators list (i.e.
  /// have been added and not yet removed).
  void removeGenerator(DefinitionGenerator &G);

  /// Set the link order to be used when fixing up definitions in JITDylib.
  /// This will replace the previous link order, and apply to any symbol
  /// resolutions made for definitions in this JITDylib after the call to
  /// setLinkOrder (even if the definition itself was added before the
  /// call).
  ///
  /// If LinkAgainstThisJITDylibFirst is true (the default) then this JITDylib
  /// will add itself to the beginning of the LinkOrder (Clients should not
  /// put this JITDylib in the list in this case, to avoid redundant lookups).
  ///
  /// If LinkAgainstThisJITDylibFirst is false then the link order will be used
  /// as-is. The primary motivation for this feature is to support deliberate
  /// shadowing of symbols in this JITDylib by a facade JITDylib. For example,
  /// the facade may resolve function names to stubs, and the stubs may compile
  /// lazily by looking up symbols in this dylib. Adding the facade dylib
  /// as the first in the link order (instead of this dylib) ensures that
  /// definitions within this dylib resolve to the lazy-compiling stubs,
  /// rather than immediately materializing the definitions in this dylib.
  void setLinkOrder(JITDylibSearchOrder NewSearchOrder,
                    bool LinkAgainstThisJITDylibFirst = true);

  /// Add the given JITDylib to the link order for definitions in this
  /// JITDylib.
  void addToLinkOrder(JITDylib &JD,
                      JITDylibLookupFlags JDLookupFlags =
                          JITDylibLookupFlags::MatchExportedSymbolsOnly);

  /// Replace OldJD with NewJD in the link order if OldJD is present.
  /// Otherwise this operation is a no-op.
  void replaceInLinkOrder(JITDylib &OldJD, JITDylib &NewJD,
                          JITDylibLookupFlags JDLookupFlags =
                              JITDylibLookupFlags::MatchExportedSymbolsOnly);

  /// Remove the given JITDylib from the link order for this JITDylib if it is
  /// present. Otherwise this operation is a no-op.
  void removeFromLinkOrder(JITDylib &JD);

  /// Do something with the link order (run under the session lock).
  template <typename Func>
  auto withLinkOrderDo(Func &&F)
      -> decltype(F(std::declval<const JITDylibSearchOrder &>()));

  /// Define all symbols provided by the materialization unit to be part of this
  /// JITDylib.
  ///
  /// If RT is not specified then the default resource tracker will be used.
  ///
  /// This overload always takes ownership of the MaterializationUnit. If any
  /// errors occur, the MaterializationUnit consumed.
  template <typename MaterializationUnitType>
  Error define(std::unique_ptr<MaterializationUnitType> &&MU,
               ResourceTrackerSP RT = nullptr);

  /// Define all symbols provided by the materialization unit to be part of this
  /// JITDylib.
  ///
  /// This overload only takes ownership of the MaterializationUnit no error is
  /// generated. If an error occurs, ownership remains with the caller. This
  /// may allow the caller to modify the MaterializationUnit to correct the
  /// issue, then re-call define.
  template <typename MaterializationUnitType>
  Error define(std::unique_ptr<MaterializationUnitType> &MU,
               ResourceTrackerSP RT = nullptr);

  /// Tries to remove the given symbols.
  ///
  /// If any symbols are not defined in this JITDylib this method will return
  /// a SymbolsNotFound error covering the missing symbols.
  ///
  /// If all symbols are found but some symbols are in the process of being
  /// materialized this method will return a SymbolsCouldNotBeRemoved error.
  ///
  /// On success, all symbols are removed. On failure, the JITDylib state is
  /// left unmodified (no symbols are removed).
  Error remove(const SymbolNameSet &Names);

  /// Dump current JITDylib state to OS.
  void dump(raw_ostream &OS);

  /// Returns the given JITDylibs and all of their transitive dependencies in
  /// DFS order (based on linkage relationships). Each JITDylib will appear
  /// only once.
  static std::vector<JITDylibSP> getDFSLinkOrder(ArrayRef<JITDylibSP> JDs);

  /// Returns the given JITDylibs and all of their transitive dependensies in
  /// reverse DFS order (based on linkage relationships). Each JITDylib will
  /// appear only once.
  static std::vector<JITDylibSP>
  getReverseDFSLinkOrder(ArrayRef<JITDylibSP> JDs);

  /// Return this JITDylib and its transitive dependencies in DFS order
  /// based on linkage relationships.
  std::vector<JITDylibSP> getDFSLinkOrder();

  /// Rteurn this JITDylib and its transitive dependencies in reverse DFS order
  /// based on linkage relationships.
  std::vector<JITDylibSP> getReverseDFSLinkOrder();

private:
  using AsynchronousSymbolQueryList =
      std::vector<std::shared_ptr<AsynchronousSymbolQuery>>;

  struct UnmaterializedInfo {
    UnmaterializedInfo(std::unique_ptr<MaterializationUnit> MU,
                       ResourceTracker *RT)
        : MU(std::move(MU)), RT(RT) {}

    std::unique_ptr<MaterializationUnit> MU;
    ResourceTracker *RT;
  };

  using UnmaterializedInfosMap =
      DenseMap<SymbolStringPtr, std::shared_ptr<UnmaterializedInfo>>;

  using UnmaterializedInfosList =
      std::vector<std::shared_ptr<UnmaterializedInfo>>;

  struct MaterializingInfo {
    SymbolDependenceMap Dependants;
    SymbolDependenceMap UnemittedDependencies;

    void addQuery(std::shared_ptr<AsynchronousSymbolQuery> Q);
    void removeQuery(const AsynchronousSymbolQuery &Q);
    AsynchronousSymbolQueryList takeQueriesMeeting(SymbolState RequiredState);
    AsynchronousSymbolQueryList takeAllPendingQueries() {
      return std::move(PendingQueries);
    }
    bool hasQueriesPending() const { return !PendingQueries.empty(); }
    const AsynchronousSymbolQueryList &pendingQueries() const {
      return PendingQueries;
    }
  private:
    AsynchronousSymbolQueryList PendingQueries;
  };

  using MaterializingInfosMap = DenseMap<SymbolStringPtr, MaterializingInfo>;

  class SymbolTableEntry {
  public:
    SymbolTableEntry() = default;
    SymbolTableEntry(JITSymbolFlags Flags)
        : Flags(Flags), State(static_cast<uint8_t>(SymbolState::NeverSearched)),
          MaterializerAttached(false), PendingRemoval(false) {}

    JITTargetAddress getAddress() const { return Addr; }
    JITSymbolFlags getFlags() const { return Flags; }
    SymbolState getState() const { return static_cast<SymbolState>(State); }

    bool hasMaterializerAttached() const { return MaterializerAttached; }
    bool isPendingRemoval() const { return PendingRemoval; }

    void setAddress(JITTargetAddress Addr) { this->Addr = Addr; }
    void setFlags(JITSymbolFlags Flags) { this->Flags = Flags; }
    void setState(SymbolState State) {
      assert(static_cast<uint8_t>(State) < (1 << 6) &&
             "State does not fit in bitfield");
      this->State = static_cast<uint8_t>(State);
    }

    void setMaterializerAttached(bool MaterializerAttached) {
      this->MaterializerAttached = MaterializerAttached;
    }

    void setPendingRemoval(bool PendingRemoval) {
      this->PendingRemoval = PendingRemoval;
    }

    JITEvaluatedSymbol getSymbol() const {
      return JITEvaluatedSymbol(Addr, Flags);
    }

  private:
    JITTargetAddress Addr = 0;
    JITSymbolFlags Flags;
    uint8_t State : 6;
    uint8_t MaterializerAttached : 1;
    uint8_t PendingRemoval : 1;
  };

  using SymbolTable = DenseMap<SymbolStringPtr, SymbolTableEntry>;

  JITDylib(ExecutionSession &ES, std::string Name);

  ResourceTrackerSP getTracker(MaterializationResponsibility &MR);
  std::pair<AsynchronousSymbolQuerySet, std::shared_ptr<SymbolDependenceMap>>
  removeTracker(ResourceTracker &RT);

  void transferTracker(ResourceTracker &DstRT, ResourceTracker &SrcRT);

  Error defineImpl(MaterializationUnit &MU);

  void installMaterializationUnit(std::unique_ptr<MaterializationUnit> MU,
                                  ResourceTracker &RT);

  void detachQueryHelper(AsynchronousSymbolQuery &Q,
                         const SymbolNameSet &QuerySymbols);

  void transferEmittedNodeDependencies(MaterializingInfo &DependantMI,
                                       const SymbolStringPtr &DependantName,
                                       MaterializingInfo &EmittedMI);

  Expected<SymbolFlagsMap> defineMaterializing(SymbolFlagsMap SymbolFlags);

  Error replace(MaterializationResponsibility &FromMR,
                std::unique_ptr<MaterializationUnit> MU);

  Expected<std::unique_ptr<MaterializationResponsibility>>
  delegate(MaterializationResponsibility &FromMR, SymbolFlagsMap SymbolFlags,
           SymbolStringPtr InitSymbol);

  SymbolNameSet getRequestedSymbols(const SymbolFlagsMap &SymbolFlags) const;

  void addDependencies(const SymbolStringPtr &Name,
                       const SymbolDependenceMap &Dependants);

  Error resolve(MaterializationResponsibility &MR, const SymbolMap &Resolved);

  Error emit(MaterializationResponsibility &MR, const SymbolFlagsMap &Emitted);

  void unlinkMaterializationResponsibility(MaterializationResponsibility &MR);

  using FailedSymbolsWorklist =
      std::vector<std::pair<JITDylib *, SymbolStringPtr>>;

  static std::pair<AsynchronousSymbolQuerySet,
                   std::shared_ptr<SymbolDependenceMap>>
      failSymbols(FailedSymbolsWorklist);

  ExecutionSession &ES;
  std::string JITDylibName;
  std::mutex GeneratorsMutex;
  bool Open = true;
  SymbolTable Symbols;
  UnmaterializedInfosMap UnmaterializedInfos;
  MaterializingInfosMap MaterializingInfos;
  std::vector<std::shared_ptr<DefinitionGenerator>> DefGenerators;
  JITDylibSearchOrder LinkOrder;
  ResourceTrackerSP DefaultTracker;

  // Map trackers to sets of symbols tracked.
  DenseMap<ResourceTracker *, SymbolNameVector> TrackerSymbols;
  DenseMap<MaterializationResponsibility *, ResourceTracker *> MRTrackers;
};

/// Platforms set up standard symbols and mediate interactions between dynamic
/// initializers (e.g. C++ static constructors) and ExecutionSession state.
/// Note that Platforms do not automatically run initializers: clients are still
/// responsible for doing this.
class Platform {
public:
  virtual ~Platform();

  /// This method will be called outside the session lock each time a JITDylib
  /// is created (unless it is created with EmptyJITDylib set) to allow the
  /// Platform to install any JITDylib specific standard symbols (e.g
  /// __dso_handle).
  virtual Error setupJITDylib(JITDylib &JD) = 0;

  /// This method will be called under the ExecutionSession lock each time a
  /// MaterializationUnit is added to a JITDylib.
  virtual Error notifyAdding(ResourceTracker &RT,
                             const MaterializationUnit &MU) = 0;

  /// This method will be called under the ExecutionSession lock when a
  /// ResourceTracker is removed.
  virtual Error notifyRemoving(ResourceTracker &RT) = 0;

  /// A utility function for looking up initializer symbols. Performs a blocking
  /// lookup for the given symbols in each of the given JITDylibs.
  static Expected<DenseMap<JITDylib *, SymbolMap>>
  lookupInitSymbols(ExecutionSession &ES,
                    const DenseMap<JITDylib *, SymbolLookupSet> &InitSyms);
};

/// An ExecutionSession represents a running JIT program.
class ExecutionSession {
  friend class InProgressLookupFlagsState;
  friend class InProgressFullLookupState;
  friend class JITDylib;
  friend class LookupState;
  friend class MaterializationResponsibility;
  friend class ResourceTracker;

public:
  /// For reporting errors.
  using ErrorReporter = std::function<void(Error)>;

  /// For dispatching MaterializationUnit::materialize calls.
  using DispatchMaterializationFunction =
      std::function<void(std::unique_ptr<MaterializationUnit> MU,
                         std::unique_ptr<MaterializationResponsibility> MR)>;

  /// Construct an ExecutionSession.
  ///
  /// SymbolStringPools may be shared between ExecutionSessions.
  ExecutionSession(std::shared_ptr<SymbolStringPool> SSP = nullptr);

  /// End the session. Closes all JITDylibs.
  Error endSession();

  /// Add a symbol name to the SymbolStringPool and return a pointer to it.
  SymbolStringPtr intern(StringRef SymName) { return SSP->intern(SymName); }

  /// Returns a shared_ptr to the SymbolStringPool for this ExecutionSession.
  std::shared_ptr<SymbolStringPool> getSymbolStringPool() const { return SSP; }

  /// Set the Platform for this ExecutionSession.
  void setPlatform(std::unique_ptr<Platform> P) { this->P = std::move(P); }

  /// Get the Platform for this session.
  /// Will return null if no Platform has been set for this ExecutionSession.
  Platform *getPlatform() { return P.get(); }

  /// Run the given lambda with the session mutex locked.
  template <typename Func> decltype(auto) runSessionLocked(Func &&F) {
    std::lock_guard<std::recursive_mutex> Lock(SessionMutex);
    return F();
  }

  /// Register the given ResourceManager with this ExecutionSession.
  /// Managers will be notified of events in reverse order of registration.
  void registerResourceManager(ResourceManager &RM);

  /// Deregister the given ResourceManager with this ExecutionSession.
  /// Manager must have been previously registered.
  void deregisterResourceManager(ResourceManager &RM);

  /// Return a pointer to the "name" JITDylib.
  /// Ownership of JITDylib remains within Execution Session
  JITDylib *getJITDylibByName(StringRef Name);

  /// Add a new bare JITDylib to this ExecutionSession.
  ///
  /// The JITDylib Name is required to be unique. Clients should verify that
  /// names are not being re-used (E.g. by calling getJITDylibByName) if names
  /// are based on user input.
  ///
  /// This call does not install any library code or symbols into the newly
  /// created JITDylib. The client is responsible for all configuration.
  JITDylib &createBareJITDylib(std::string Name);

  /// Add a new JITDylib to this ExecutionSession.
  ///
  /// The JITDylib Name is required to be unique. Clients should verify that
  /// names are not being re-used (e.g. by calling getJITDylibByName) if names
  /// are based on user input.
  ///
  /// If a Platform is attached then Platform::setupJITDylib will be called to
  /// install standard platform symbols (e.g. standard library interposes).
  /// If no Platform is attached this call is equivalent to createBareJITDylib.
  Expected<JITDylib &> createJITDylib(std::string Name);

  /// Set the error reporter function.
  ExecutionSession &setErrorReporter(ErrorReporter ReportError) {
    this->ReportError = std::move(ReportError);
    return *this;
  }

  /// Report a error for this execution session.
  ///
  /// Unhandled errors can be sent here to log them.
  void reportError(Error Err) { ReportError(std::move(Err)); }

  /// Set the materialization dispatch function.
  ExecutionSession &setDispatchMaterialization(
      DispatchMaterializationFunction DispatchMaterialization) {
    this->DispatchMaterialization = std::move(DispatchMaterialization);
    return *this;
  }

  /// Search the given JITDylibs to find the flags associated with each of the
  /// given symbols.
  void lookupFlags(LookupKind K, JITDylibSearchOrder SearchOrder,
                   SymbolLookupSet Symbols,
                   unique_function<void(Expected<SymbolFlagsMap>)> OnComplete);

  /// Blocking version of lookupFlags.
  Expected<SymbolFlagsMap> lookupFlags(LookupKind K,
                                       JITDylibSearchOrder SearchOrder,
                                       SymbolLookupSet Symbols);

  /// Search the given JITDylibs for the given symbols.
  ///
  /// SearchOrder lists the JITDylibs to search. For each dylib, the associated
  /// boolean indicates whether the search should match against non-exported
  /// (hidden visibility) symbols in that dylib (true means match against
  /// non-exported symbols, false means do not match).
  ///
  /// The NotifyComplete callback will be called once all requested symbols
  /// reach the required state.
  ///
  /// If all symbols are found, the RegisterDependencies function will be called
  /// while the session lock is held. This gives clients a chance to register
  /// dependencies for on the queried symbols for any symbols they are
  /// materializing (if a MaterializationResponsibility instance is present,
  /// this can be implemented by calling
  /// MaterializationResponsibility::addDependencies). If there are no
  /// dependenant symbols for this query (e.g. it is being made by a top level
  /// client to get an address to call) then the value NoDependenciesToRegister
  /// can be used.
  void lookup(LookupKind K, const JITDylibSearchOrder &SearchOrder,
              SymbolLookupSet Symbols, SymbolState RequiredState,
              SymbolsResolvedCallback NotifyComplete,
              RegisterDependenciesFunction RegisterDependencies);

  /// Blocking version of lookup above. Returns the resolved symbol map.
  /// If WaitUntilReady is true (the default), will not return until all
  /// requested symbols are ready (or an error occurs). If WaitUntilReady is
  /// false, will return as soon as all requested symbols are resolved,
  /// or an error occurs. If WaitUntilReady is false and an error occurs
  /// after resolution, the function will return a success value, but the
  /// error will be reported via reportErrors.
  Expected<SymbolMap> lookup(const JITDylibSearchOrder &SearchOrder,
                             const SymbolLookupSet &Symbols,
                             LookupKind K = LookupKind::Static,
                             SymbolState RequiredState = SymbolState::Ready,
                             RegisterDependenciesFunction RegisterDependencies =
                                 NoDependenciesToRegister);

  /// Convenience version of blocking lookup.
  /// Searches each of the JITDylibs in the search order in turn for the given
  /// symbol.
  Expected<JITEvaluatedSymbol>
  lookup(const JITDylibSearchOrder &SearchOrder, SymbolStringPtr Symbol,
         SymbolState RequiredState = SymbolState::Ready);

  /// Convenience version of blocking lookup.
  /// Searches each of the JITDylibs in the search order in turn for the given
  /// symbol. The search will not find non-exported symbols.
  Expected<JITEvaluatedSymbol>
  lookup(ArrayRef<JITDylib *> SearchOrder, SymbolStringPtr Symbol,
         SymbolState RequiredState = SymbolState::Ready);

  /// Convenience version of blocking lookup.
  /// Searches each of the JITDylibs in the search order in turn for the given
  /// symbol. The search will not find non-exported symbols.
  Expected<JITEvaluatedSymbol>
  lookup(ArrayRef<JITDylib *> SearchOrder, StringRef Symbol,
         SymbolState RequiredState = SymbolState::Ready);

  /// Materialize the given unit.
  void
  dispatchMaterialization(std::unique_ptr<MaterializationUnit> MU,
                          std::unique_ptr<MaterializationResponsibility> MR) {
    assert(MU && "MU must be non-null");
    DEBUG_WITH_TYPE("orc", dumpDispatchInfo(MR->getTargetJITDylib(), *MU));
    DispatchMaterialization(std::move(MU), std::move(MR));
  }

  /// Dump the state of all the JITDylibs in this session.
  void dump(raw_ostream &OS);

private:
  static void logErrorsToStdErr(Error Err) {
    logAllUnhandledErrors(std::move(Err), errs(), "JIT session error: ");
  }

  static void materializeOnCurrentThread(
      std::unique_ptr<MaterializationUnit> MU,
      std::unique_ptr<MaterializationResponsibility> MR) {
    MU->materialize(std::move(MR));
  }

  void dispatchOutstandingMUs();

  static std::unique_ptr<MaterializationResponsibility>
  createMaterializationResponsibility(ResourceTracker &RT,
                                      SymbolFlagsMap Symbols,
                                      SymbolStringPtr InitSymbol) {
    auto &JD = RT.getJITDylib();
    std::unique_ptr<MaterializationResponsibility> MR(
        new MaterializationResponsibility(&JD, std::move(Symbols),
                                          std::move(InitSymbol)));
    JD.MRTrackers[MR.get()] = &RT;
    return MR;
  }

  Error removeResourceTracker(ResourceTracker &RT);
  void transferResourceTracker(ResourceTracker &DstRT, ResourceTracker &SrcRT);
  void destroyResourceTracker(ResourceTracker &RT);

  // State machine functions for query application..

  /// IL_updateCandidatesFor is called to remove already-defined symbols that
  /// match a given query from the set of candidate symbols to generate
  /// definitions for (no need to generate a definition if one already exists).
  Error IL_updateCandidatesFor(JITDylib &JD, JITDylibLookupFlags JDLookupFlags,
                               SymbolLookupSet &Candidates,
                               SymbolLookupSet *NonCandidates);

  /// OL_applyQueryPhase1 is an optionally re-startable loop for triggering
  /// definition generation. It is called when a lookup is performed, and again
  /// each time that LookupState::continueLookup is called.
  void OL_applyQueryPhase1(std::unique_ptr<InProgressLookupState> IPLS,
                           Error Err);

  /// OL_completeLookup is run once phase 1 successfully completes for a lookup
  /// call. It attempts to attach the symbol to all symbol table entries and
  /// collect all MaterializationUnits to dispatch. If this method fails then
  /// all MaterializationUnits will be left un-materialized.
  void OL_completeLookup(std::unique_ptr<InProgressLookupState> IPLS,
                         std::shared_ptr<AsynchronousSymbolQuery> Q,
                         RegisterDependenciesFunction RegisterDependencies);

  /// OL_completeLookupFlags is run once phase 1 successfully completes for a
  /// lookupFlags call.
  void OL_completeLookupFlags(
      std::unique_ptr<InProgressLookupState> IPLS,
      unique_function<void(Expected<SymbolFlagsMap>)> OnComplete);

  // State machine functions for MaterializationResponsibility.
  void OL_destroyMaterializationResponsibility(
      MaterializationResponsibility &MR);
  SymbolNameSet OL_getRequestedSymbols(const MaterializationResponsibility &MR);
  Error OL_notifyResolved(MaterializationResponsibility &MR,
                          const SymbolMap &Symbols);
  Error OL_notifyEmitted(MaterializationResponsibility &MR);
  Error OL_defineMaterializing(MaterializationResponsibility &MR,
                               SymbolFlagsMap SymbolFlags);
  void OL_notifyFailed(MaterializationResponsibility &MR);
  Error OL_replace(MaterializationResponsibility &MR,
                   std::unique_ptr<MaterializationUnit> MU);
  Expected<std::unique_ptr<MaterializationResponsibility>>
  OL_delegate(MaterializationResponsibility &MR, const SymbolNameSet &Symbols);
  void OL_addDependencies(MaterializationResponsibility &MR,
                          const SymbolStringPtr &Name,
                          const SymbolDependenceMap &Dependencies);
  void OL_addDependenciesForAll(MaterializationResponsibility &MR,
                                const SymbolDependenceMap &Dependencies);

#ifndef NDEBUG
  void dumpDispatchInfo(JITDylib &JD, MaterializationUnit &MU);
#endif // NDEBUG

  mutable std::recursive_mutex SessionMutex;
  bool SessionOpen = true;
  std::shared_ptr<SymbolStringPool> SSP;
  std::unique_ptr<Platform> P;
  ErrorReporter ReportError = logErrorsToStdErr;
  DispatchMaterializationFunction DispatchMaterialization =
      materializeOnCurrentThread;

  std::vector<ResourceManager *> ResourceManagers;

  std::vector<JITDylibSP> JDs;

  // FIXME: Remove this (and runOutstandingMUs) once the linking layer works
  //        with callbacks from asynchronous queries.
  mutable std::recursive_mutex OutstandingMUsMutex;
  std::vector<std::pair<std::unique_ptr<MaterializationUnit>,
                        std::unique_ptr<MaterializationResponsibility>>>
      OutstandingMUs;
};

inline ExecutionSession &MaterializationResponsibility::getExecutionSession() {
  return JD->getExecutionSession();
}

template <typename Func>
Error MaterializationResponsibility::withResourceKeyDo(Func &&F) const {
  return JD->getExecutionSession().runSessionLocked([&]() -> Error {
    auto I = JD->MRTrackers.find(this);
    assert(I != JD->MRTrackers.end() && "No tracker for this MR");
    if (I->second->isDefunct())
      return make_error<ResourceTrackerDefunct>(I->second);
    F(I->second->getKeyUnsafe());
    return Error::success();
  });
}

template <typename GeneratorT>
GeneratorT &JITDylib::addGenerator(std::unique_ptr<GeneratorT> DefGenerator) {
  auto &G = *DefGenerator;
  std::lock_guard<std::mutex> Lock(GeneratorsMutex);
  DefGenerators.push_back(std::move(DefGenerator));
  return G;
}

template <typename Func>
auto JITDylib::withLinkOrderDo(Func &&F)
    -> decltype(F(std::declval<const JITDylibSearchOrder &>())) {
  return ES.runSessionLocked([&]() { return F(LinkOrder); });
}

template <typename MaterializationUnitType>
Error JITDylib::define(std::unique_ptr<MaterializationUnitType> &&MU,
                       ResourceTrackerSP RT) {
  assert(MU && "Can not define with a null MU");

  if (MU->getSymbols().empty()) {
    // Empty MUs are allowable but pathological, so issue a warning.
    DEBUG_WITH_TYPE("orc", {
      dbgs() << "Warning: Discarding empty MU " << MU->getName() << " for "
             << getName() << "\n";
    });
    return Error::success();
  } else
    DEBUG_WITH_TYPE("orc", {
      dbgs() << "Defining MU " << MU->getName() << " for " << getName()
             << " (tracker: ";
      if (RT == getDefaultResourceTracker())
        dbgs() << "default)";
      else if (RT)
        dbgs() << RT.get() << ")\n";
      else
        dbgs() << "0x0, default will be used)\n";
    });

  return ES.runSessionLocked([&, this]() -> Error {
    if (auto Err = defineImpl(*MU))
      return Err;

    if (!RT)
      RT = getDefaultResourceTracker();

    if (auto *P = ES.getPlatform()) {
      if (auto Err = P->notifyAdding(*RT, *MU))
        return Err;
    }

    installMaterializationUnit(std::move(MU), *RT);
    return Error::success();
  });
}

template <typename MaterializationUnitType>
Error JITDylib::define(std::unique_ptr<MaterializationUnitType> &MU,
                       ResourceTrackerSP RT) {
  assert(MU && "Can not define with a null MU");

  if (MU->getSymbols().empty()) {
    // Empty MUs are allowable but pathological, so issue a warning.
    DEBUG_WITH_TYPE("orc", {
      dbgs() << "Warning: Discarding empty MU " << MU->getName() << getName()
             << "\n";
    });
    return Error::success();
  } else
    DEBUG_WITH_TYPE("orc", {
      dbgs() << "Defining MU " << MU->getName() << " for " << getName()
             << " (tracker: ";
      if (RT == getDefaultResourceTracker())
        dbgs() << "default)";
      else if (RT)
        dbgs() << RT.get() << ")\n";
      else
        dbgs() << "0x0, default will be used)\n";
    });

  return ES.runSessionLocked([&, this]() -> Error {
    if (auto Err = defineImpl(*MU))
      return Err;

    if (!RT)
      RT = getDefaultResourceTracker();

    if (auto *P = ES.getPlatform()) {
      if (auto Err = P->notifyAdding(*RT, *MU))
        return Err;
    }

    installMaterializationUnit(std::move(MU), *RT);
    return Error::success();
  });
}

/// ReexportsGenerator can be used with JITDylib::addGenerator to automatically
/// re-export a subset of the source JITDylib's symbols in the target.
class ReexportsGenerator : public DefinitionGenerator {
public:
  using SymbolPredicate = std::function<bool(SymbolStringPtr)>;

  /// Create a reexports generator. If an Allow predicate is passed, only
  /// symbols for which the predicate returns true will be reexported. If no
  /// Allow predicate is passed, all symbols will be exported.
  ReexportsGenerator(JITDylib &SourceJD,
                     JITDylibLookupFlags SourceJDLookupFlags,
                     SymbolPredicate Allow = SymbolPredicate());

  Error tryToGenerate(LookupState &LS, LookupKind K, JITDylib &JD,
                      JITDylibLookupFlags JDLookupFlags,
                      const SymbolLookupSet &LookupSet) override;

private:
  JITDylib &SourceJD;
  JITDylibLookupFlags SourceJDLookupFlags;
  SymbolPredicate Allow;
};

// --------------- IMPLEMENTATION --------------
// Implementations for inline functions/methods.
// ---------------------------------------------

inline MaterializationResponsibility::~MaterializationResponsibility() {
  JD->getExecutionSession().OL_destroyMaterializationResponsibility(*this);
}

inline SymbolNameSet MaterializationResponsibility::getRequestedSymbols() const {
  return JD->getExecutionSession().OL_getRequestedSymbols(*this);
}

inline Error MaterializationResponsibility::notifyResolved(
    const SymbolMap &Symbols) {
  return JD->getExecutionSession().OL_notifyResolved(*this, Symbols);
}

inline Error MaterializationResponsibility::notifyEmitted() {
  return JD->getExecutionSession().OL_notifyEmitted(*this);
}

inline Error MaterializationResponsibility::defineMaterializing(
    SymbolFlagsMap SymbolFlags) {
  return JD->getExecutionSession().OL_defineMaterializing(
      *this, std::move(SymbolFlags));
}

inline void MaterializationResponsibility::failMaterialization() {
  JD->getExecutionSession().OL_notifyFailed(*this);
}

inline Error MaterializationResponsibility::replace(
    std::unique_ptr<MaterializationUnit> MU) {
  return JD->getExecutionSession().OL_replace(*this, std::move(MU));
}

inline Expected<std::unique_ptr<MaterializationResponsibility>>
MaterializationResponsibility::delegate(const SymbolNameSet &Symbols) {
  return JD->getExecutionSession().OL_delegate(*this, Symbols);
}

inline void MaterializationResponsibility::addDependencies(
    const SymbolStringPtr &Name, const SymbolDependenceMap &Dependencies) {
  JD->getExecutionSession().OL_addDependencies(*this, Name, Dependencies);
}

inline void MaterializationResponsibility::addDependenciesForAll(
    const SymbolDependenceMap &Dependencies) {
  JD->getExecutionSession().OL_addDependenciesForAll(*this, Dependencies);
}

} // End namespace orc
} // End namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_CORE_H
