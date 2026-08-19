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
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/RWMutex.h"
#include <cstdint>
#include <optional>
#include <vector>

namespace mlir {
class MLIRContext;
namespace detail {

/// Per-context store of the "may transitively contain a SymbolRefAttr" fact for
/// uniqued types and attributes. Two tables mirror the context's two uniquers,
/// so type and attribute opaque pointers never need to be argued disjoint.
///
/// Each table is a packed open-addressed array of single-word slots. A live
/// slot holds the uniqued opaque pointer with its answer in bit 0; both storage
/// families are 8-aligned, so bit 0 is always free to carry the fact (the
/// static_asserts below stand watch over that). A zero word is an empty slot,
/// which a live entry can never collide with because a uniqued pointer is never
/// null.
///
/// Locking discipline: the tables live entirely behind one SmartRWMutex. A
/// lookup holds the read lock for its whole probe; an insert -- including any
/// table growth it triggers -- holds the write lock. Readers therefore never
/// observe a half-grown table and the plain arrays need no per-slot atomics.
/// The `lock` flag threaded through each operation is the context's runtime
/// multithreading flag: when it is false the store is touched single-threaded
/// and no lock is taken, mirroring MLIRContext's ScopedWriterLock.
///
/// Probe contract: a lookup probes forward from the mixed home slot until it
/// meets the key (a hit) or an empty slot (a miss); it is never bounded short
/// of an empty slot. Growth keeps the load factor at or below one half, so an
/// empty slot always exists and every probe terminates. Insert places a key at
/// the first empty slot on its probe, or returns the resident answer if the key
/// is already present, so a racing duplicate fill is a no-op.
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
  // Both uniqued-storage families are 8-aligned, so bit 0 of a stored pointer
  // is free to hold the cached answer.
  static_assert(
      alignof(TypeStorage) >= 8,
      "type storage must be 8-aligned so bit 0 of its pointer is free "
      "to tag the cached answer");
  static_assert(
      alignof(AttributeStorage) >= 8,
      "attribute storage must be 8-aligned so bit 0 of its pointer is "
      "free to tag the cached answer");

  /// One packed open-addressed table of tagged-pointer facts. It is not
  /// self-synchronizing: the enclosing cache serializes all access with its
  /// SmartRWMutex, so every method here runs under the read lock (lookup) or
  /// the write lock (insert and the growth it may trigger).
  struct Table {
    // Power-of-two length, or empty before the first insert. 0 == empty slot;
    // a live slot is (uniqued pointer | answer bit).
    std::vector<uintptr_t> slots;
    size_t count = 0;

    std::optional<bool> lookup(uintptr_t key) const {
      if (slots.empty())
        return std::nullopt;
      unsigned shift = 64u - llvm::Log2_64(slots.size());
      size_t mask = slots.size() - 1;
      for (size_t i = home(key, shift);; i = (i + 1) & mask) {
        uintptr_t word = slots[i];
        if (word == 0)
          return std::nullopt;
        if ((word & ~static_cast<uintptr_t>(1)) == key)
          return static_cast<bool>(word & 1);
      }
    }

    bool insert(uintptr_t key, bool value) {
      // Grow before this insert could push the load past one half, so a lookup
      // is always guaranteed an empty slot to terminate on.
      if (slots.empty() || (count + 1) * 2 > slots.size())
        grow();
      for (;;) {
        unsigned shift = 64u - llvm::Log2_64(slots.size());
        size_t mask = slots.size() - 1;
        size_t probe = 0;
        for (size_t i = home(key, shift);; i = (i + 1) & mask) {
          uintptr_t word = slots[i];
          if (word == 0) {
            slots[i] = key | static_cast<uintptr_t>(value);
            ++count;
            return value;
          }
          if ((word & ~static_cast<uintptr_t>(1)) == key)
            return static_cast<bool>(word & 1);
          // A cluster longer than the bound is a sign the mix has degenerated
          // for this address run; grow to break it up rather than lengthen it.
          if (++probe > kMaxProbe)
            break;
        }
        grow();
      }
    }

  private:
    // Fibonacci hashing: one multiply by the 64-bit golden-ratio constant, then
    // take the top log2(capacity) bits. Bump-allocated storage pointers advance
    // in regular strides, which a bare mask would fold into long clusters; the
    // multiply spreads those strides across the whole table.
    static size_t home(uintptr_t key, unsigned shift) {
      uint64_t mixed = static_cast<uint64_t>(key >> 3) * 0x9E3779B97F4A7C15ULL;
      return static_cast<size_t>(mixed >> shift);
    }

    void grow() {
      size_t newCapacity = slots.empty() ? kInitialCapacity : slots.size() * 2;
      std::vector<uintptr_t> old = std::move(slots);
      slots.assign(newCapacity, 0);
      unsigned shift = 64u - llvm::Log2_64(newCapacity);
      size_t mask = newCapacity - 1;
      for (uintptr_t word : old) {
        if (word == 0)
          continue;
        size_t i = home(word & ~static_cast<uintptr_t>(1), shift);
        while (slots[i] != 0)
          i = (i + 1) & mask;
        slots[i] = word;
      }
    }

    static constexpr size_t kInitialCapacity = 8;
    static constexpr size_t kMaxProbe = 16;
  };

  std::optional<bool> lookupImpl(const Table &table, const void *key,
                                 bool lock) const {
    std::optional<llvm::sys::SmartScopedReader<true>> guard;
    if (lock)
      guard.emplace(mutex);
    return table.lookup(reinterpret_cast<uintptr_t>(key));
  }

  bool insertImpl(Table &table, const void *key, bool value, bool lock) {
    std::optional<llvm::sys::SmartScopedWriter<true>> guard;
    if (lock)
      guard.emplace(mutex);
    return table.insert(reinterpret_cast<uintptr_t>(key), value);
  }

  mutable llvm::sys::SmartRWMutex<true> mutex;
  Table typeEntries;
  Table attrEntries;
};

/// Return the given context's symbol-reference containment cache. Defined in
/// MLIRContext.cpp, where MLIRContextImpl is visible.
SymbolRefContainmentCache &getSymbolRefContainmentCache(MLIRContext *ctx);

} // namespace detail
} // namespace mlir

#endif // MLIR_LIB_IR_SYMBOLREFCONTAINMENTCACHE_H
