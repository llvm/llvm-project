//===- SymbolRefContainmentCache.h - Symbol-ref containment cache ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A context-owned store recording, per uniqued type and attribute, whether it
// may transitively contain a SymbolRefAttr. Symbol-table verification consults
// it to prune the symbol-use walk.
//
// The store is filled lazily and never invalidated. Uniqued storage is immortal
// and immutable for the context's lifetime, so a cached answer can never go
// stale, and a pointer can never be recycled to alias another object within one
// context. Every entry is write-once: false only for a provably
// reference-free immutable subtree, true for everything else, including
// mutable-storage kinds, whose contents the fill never reads.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_IR_SYMBOLREFCONTAINMENTCACHE_H
#define MLIR_LIB_IR_SYMBOLREFCONTAINMENTCACHE_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/RWMutex.h"
#include <optional>

namespace mlir {
class MLIRContext;
namespace detail {

/// Per-context store of the "may transitively contain a SymbolRefAttr" fact for
/// uniqued types and attributes. Two maps mirror the context's two uniquers, so
/// type and attribute opaque pointers never need to be argued disjoint. The
/// `lock` flag threaded through each operation is the context's runtime
/// multithreading flag: when it is false the store is touched single-threaded
/// and no lock is taken, mirroring MLIRContext's ScopedWriterLock.
class SymbolRefContainmentCache {
public:
  SymbolRefContainmentCache() = default;
  SymbolRefContainmentCache(const SymbolRefContainmentCache &) = delete;
  SymbolRefContainmentCache &
  operator=(const SymbolRefContainmentCache &) = delete;

  /// Return the cached answer for `type`/`attr`, or nullopt if not yet filled.
  std::optional<bool> lookup(Type type, bool lock) const {
    return lookupImpl(typeEntries, type.getAsOpaquePointer(), lock);
  }
  std::optional<bool> lookup(Attribute attr, bool lock) const {
    return lookupImpl(attrEntries, attr.getAsOpaquePointer(), lock);
  }

  /// Record `value` for `type`/`attr` if no entry exists yet and return the
  /// resident answer. A racing duplicate fill computes the same immutable fact,
  /// so the losing insert is a no-op.
  bool insert(Type type, bool value, bool lock) {
    return insertImpl(typeEntries, type.getAsOpaquePointer(), value, lock);
  }
  bool insert(Attribute attr, bool value, bool lock) {
    return insertImpl(attrEntries, attr.getAsOpaquePointer(), value, lock);
  }

private:
  using Map = DenseMap<const void *, bool>;

  std::optional<bool> lookupImpl(const Map &map, const void *key,
                                 bool lock) const {
    std::optional<llvm::sys::SmartScopedReader<true>> guard;
    if (lock)
      guard.emplace(mutex);
    auto it = map.find(key);
    if (it == map.end())
      return std::nullopt;
    return it->second;
  }

  bool insertImpl(Map &map, const void *key, bool value, bool lock) {
    std::optional<llvm::sys::SmartScopedWriter<true>> guard;
    if (lock)
      guard.emplace(mutex);
    return map.try_emplace(key, value).first->second;
  }

  mutable llvm::sys::SmartRWMutex<true> mutex;
  Map typeEntries;
  Map attrEntries;
};

/// Return the given context's symbol-reference containment cache. Defined in
/// MLIRContext.cpp, where MLIRContextImpl is visible.
SymbolRefContainmentCache &getSymbolRefContainmentCache(MLIRContext *ctx);

} // namespace detail
} // namespace mlir

#endif // MLIR_LIB_IR_SYMBOLREFCONTAINMENTCACHE_H
