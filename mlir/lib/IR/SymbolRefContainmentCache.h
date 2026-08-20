//===- SymbolRefContainmentCache.h - Symbol-ref containment cache ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A context-owned cache answering, for a uniqued type or attribute, whether it
// may transitively contain a SymbolRefAttr. Symbol-table verification consults
// it to prune its symbol-use walk to the subtrees that can carry a symbol use.
//
// A recorded answer is a pure function of immutable structure --
// mutable-storage kinds are never read or recorded -- so it never goes stale.
// Outside a transient scope, uniqued storage is also immortal, so two live
// objects can never share an address. A transient scope frees the storage it
// allocated when it ends; the context clears this cache at that point, before
// any freed address can be recycled into a different object that would inherit
// a stale answer. One DenseSet of opaque pointers therefore keys both kinds
// without arguing them disjoint -- the attribute side already pools two
// allocators, the uniquer and the DistinctAttr allocator, on that same
// argument.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_IR_SYMBOLREFCONTAINMENTCACHE_H
#define MLIR_LIB_IR_SYMBOLREFCONTAINMENTCACHE_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/RWMutex.h"
#include "llvm/Support/Threading.h"
#include <optional>

namespace mlir {
class MLIRContext;
namespace detail {

/// Per-context cache of the "provably free of a transitive SymbolRefAttr" fact
/// for uniqued types and attributes, computed on demand, memoizing proven-free
/// answers. A true answer is authoritative -- the object references no symbol,
/// so a conforming SymbolUserTypeInterface has a vacuous verifySymbolUses and
/// the caller may prune it unvisited. A false answer is conservative: the
/// object may contain a reference and the caller walks it, and a
/// mutable-storage kind, whose sub-elements may change after the query, and
/// anything containing one, always answers false.
class SymbolRefContainmentCache {
public:
  /// `isMultithreaded` is the context's runtime multithreading flag, read live
  /// on every probe to decide whether the set needs its lock; it must outlive
  /// this cache.
  explicit SymbolRefContainmentCache(const bool &isMultithreaded)
      : isMultithreaded(isMultithreaded) {}
  SymbolRefContainmentCache(const SymbolRefContainmentCache &) = delete;
  SymbolRefContainmentCache &
  operator=(const SymbolRefContainmentCache &) = delete;

  /// Return whether `type`/`attr` is provably free of any transitive
  /// SymbolRefAttr -- true is safe to prune, false is walked conservatively.
  bool isSymbolRefFree(Type type) { return !compute(type); }
  bool isSymbolRefFree(Attribute attr) { return !compute(attr); }

  /// Drop every recorded fact. The cache is a pure memo of proven-clear
  /// objects, so forgetting them all is always correct and only costs a
  /// recomputation on next encounter. The context calls this when a transient
  /// scope ends and its transiently allocated storage is freed, before any
  /// address can be recycled into a different object that would inherit a stale
  /// answer.
  void clear() {
    std::optional<llvm::sys::SmartScopedWriter<true>> guard;
    if (shouldLock())
      guard.emplace(mutex);
    clearSet.clear();
  }

private:
  // A type is never itself a SymbolRefAttr; an attribute is one exactly when
  // isa<SymbolRefAttr> holds (covering FlatSymbolRefAttr).
  static bool isSelfSymbolRef(Type) { return false; }
  static bool isSelfSymbolRef(Attribute attr) {
    return isa<SymbolRefAttr>(attr);
  }

  /// Return whether `obj` may transitively contain a SymbolRefAttr, recording a
  /// proven-clear result so it need never recompute. The recursion needs no
  /// in-progress guard: immutable objects form a DAG (sub-elements are interned
  /// before their parents), so every cycle passes through a mutable kind, where
  /// the descent stops above before reading contents. Once one sub-element
  /// answers may-contain the rest need not be visited: walkImmediateSubElements
  /// cannot be interrupted, but the callback is a no-op once `mayContain`
  /// holds.
  template <typename T>
  bool compute(T obj) {
    const void *key = obj.getAsOpaquePointer();
    if (isKnownClear(key))
      return false;
    bool mayContain = isSelfSymbolRef(obj) ||
                      obj.template hasTrait<StorageUserTrait::IsMutable>();
    if (!mayContain) {
      auto walkSub = [&](auto sub) {
        if (!mayContain && sub)
          mayContain = compute(sub);
      };
      obj.walkImmediateSubElements(walkSub, walkSub);
    }
    if (!mayContain)
      markClear(key);
    return mayContain;
  }

  bool isKnownClear(const void *key) const {
    std::optional<llvm::sys::SmartScopedReader<true>> guard;
    if (shouldLock())
      guard.emplace(mutex);
    return clearSet.contains(key);
  }
  void markClear(const void *key) {
    std::optional<llvm::sys::SmartScopedWriter<true>> guard;
    if (shouldLock())
      guard.emplace(mutex);
    clearSet.insert(key);
  }

  /// The set is serialized only while the context runs multithreaded;
  /// llvm_is_multithreaded() is a compile-time constant folded in here.
  bool shouldLock() const {
    return isMultithreaded && llvm::llvm_is_multithreaded();
  }

  const bool &isMultithreaded;
  // The set lives behind this lock: a read lock guards the contains probe, a
  // write lock the insert and any growth it triggers, and neither is ever held
  // across the recursion.
  mutable llvm::sys::SmartRWMutex<true> mutex;
  // Only the proven-clear opaque pointers are recorded; a may-contain object --
  // including every mutable-storage kind, whose contents are never read -- is
  // left out and recomputed on each encounter, so no fact that could later go
  // stale is ever stored.
  llvm::DenseSet<const void *> clearSet;
};

/// Return the given context's symbol-reference containment cache. Defined in
/// MLIRContext.cpp, where MLIRContextImpl is visible.
SymbolRefContainmentCache &getSymbolRefContainmentCache(MLIRContext *ctx);

} // namespace detail
} // namespace mlir

#endif // MLIR_LIB_IR_SYMBOLREFCONTAINMENTCACHE_H
