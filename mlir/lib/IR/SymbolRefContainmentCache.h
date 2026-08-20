//===- SymbolRefContainmentCache.h - Symbol-ref containment cache ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A context-owned store recording which uniqued types and attributes are
// provably free of a transitive SymbolRefAttr. Symbol-table verification
// consults it to prune the symbol-use walk.
//
// The store is filled lazily and never invalidated. Uniqued storage is immortal
// and immutable for the context's lifetime, so a recorded answer can never go
// stale, and a pointer can never be recycled to alias another object within one
// context. Only the "provably reference-free" fact is recorded; a may-contain
// answer -- including every mutable-storage kind, whose contents the fill never
// reads -- is left unrecorded and recomputed on each encounter.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_IR_SYMBOLREFCONTAINMENTCACHE_H
#define MLIR_LIB_IR_SYMBOLREFCONTAINMENTCACHE_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/RWMutex.h"
#include <optional>

namespace mlir {
class MLIRContext;
namespace detail {

/// Per-context store of the "provably free of a transitive SymbolRefAttr" fact
/// for uniqued types and attributes. Two sets mirror the context's two
/// uniquers, so type and attribute opaque pointers never need to be argued
/// disjoint.
///
/// Each set holds only the opaque pointers of objects proven reference-free.
/// Membership means "clear" (answer false); non-membership means "not yet
/// proven clear", which the fill treats as unfilled and may-contain both at
/// once -- it recomputes containment, and a genuine may-contain object is
/// recomputed cheaply because the walk early-exits at its first SymbolRefAttr.
/// A may-contain answer is therefore never stored, so a mutable-storage kind is
/// conservatively may-contain forever with no special handling.
///
/// Locking discipline: the sets live entirely behind one SmartRWMutex. A lookup
/// holds the read lock for its whole probe; an insert -- including any set
/// growth it triggers -- holds the write lock. Readers therefore never observe
/// a half-grown set and the plain sets need no per-slot atomics. The `lock`
/// flag threaded through each operation is the context's runtime multithreading
/// flag: when it is false the store is touched single-threaded and no lock is
/// taken, mirroring MLIRContext's ScopedWriterLock.
class SymbolRefContainmentCache {
public:
  SymbolRefContainmentCache() = default;
  SymbolRefContainmentCache(const SymbolRefContainmentCache &) = delete;
  SymbolRefContainmentCache &
  operator=(const SymbolRefContainmentCache &) = delete;

  /// Return false for `type`/`attr` if it is recorded clear, or nullopt if it
  /// is not yet proven clear (unfilled, treated as may-contain by the caller).
  /// A true answer is never recorded, so lookup never returns true.
  std::optional<bool> lookup(Type type, bool lock) const {
    return lookupImpl(typeEntries, type.getAsOpaquePointer(), lock);
  }
  std::optional<bool> lookup(Attribute attr, bool lock) const {
    return lookupImpl(attrEntries, attr.getAsOpaquePointer(), lock);
  }

  /// Record `type`/`attr` as clear when `value` is false and return the answer
  /// unchanged. A may-contain (`value` true) fact is not stored -- it is
  /// recomputed cheaply on each encounter -- so insert then returns true
  /// without touching the set. A racing duplicate clear fill records the same
  /// membership and is a no-op.
  bool insert(Type type, bool value, bool lock) {
    return insertImpl(typeEntries, type.getAsOpaquePointer(), value, lock);
  }
  bool insert(Attribute attr, bool value, bool lock) {
    return insertImpl(attrEntries, attr.getAsOpaquePointer(), value, lock);
  }

private:
  /// One uniquer's clear-object set. It is not self-synchronizing: the
  /// enclosing cache serializes all access with its SmartRWMutex, so every
  /// method here runs under the read lock (lookup) or the write lock (insert
  /// and the growth it may trigger).
  struct ClearSet {
    llvm::DenseSet<const void *> clear;

    std::optional<bool> lookup(const void *key) const {
      if (clear.count(key))
        return false;
      return std::nullopt;
    }

    bool insert(const void *key, bool value) {
      // Only the clear fact is durable; a may-contain answer is left out to be
      // recomputed cheaply on each encounter. A racing duplicate clear fill
      // records the same membership and is a no-op.
      if (!value)
        clear.insert(key);
      return value;
    }
  };

  std::optional<bool> lookupImpl(const ClearSet &set, const void *key,
                                 bool lock) const {
    std::optional<llvm::sys::SmartScopedReader<true>> guard;
    if (lock)
      guard.emplace(mutex);
    return set.lookup(key);
  }

  bool insertImpl(ClearSet &set, const void *key, bool value, bool lock) {
    std::optional<llvm::sys::SmartScopedWriter<true>> guard;
    if (lock)
      guard.emplace(mutex);
    return set.insert(key, value);
  }

  mutable llvm::sys::SmartRWMutex<true> mutex;
  ClearSet typeEntries;
  ClearSet attrEntries;
};

/// Return the given context's symbol-reference containment cache. Defined in
/// MLIRContext.cpp, where MLIRContextImpl is visible.
SymbolRefContainmentCache &getSymbolRefContainmentCache(MLIRContext *ctx);

} // namespace detail
} // namespace mlir

#endif // MLIR_LIB_IR_SYMBOLREFCONTAINMENTCACHE_H
