//===- SymbolStringPool.h - Multi-threaded pool for JIT symbols -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains a multi-threaded string pool suitable for use with ORC.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_SYMBOLSTRINGPOOL_H
#define LLVM_EXECUTIONENGINE_ORC_SYMBOLSTRINGPOOL_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include <atomic>
#include <mutex>

namespace llvm {

class raw_ostream;

namespace orc {

class SymbolStringPtr;

/// String pool for symbol names used by the JIT.
class SymbolStringPool {
  friend class SymbolStringPtr;

  // Implemented in DebugUtils.h.
  friend raw_ostream &operator<<(raw_ostream &OS, const SymbolStringPool &SSP);

public:
  /// Destroy a SymbolStringPool.
  ~SymbolStringPool();

  /// Create a symbol string pointer from the given string.
  SymbolStringPtr intern(StringRef S);

  /// Remove from the pool any entries that are no longer referenced.
  void clearDeadEntries();

  /// Returns true if the pool is empty.
  bool empty() const;
private:
  using RefCountType = std::atomic<size_t>;
  using PoolMap = StringMap<RefCountType>;
  using PoolMapEntry = StringMapEntry<RefCountType>;
  mutable std::mutex PoolMutex;
  PoolMap Pool;
};

/// Pointer to a pooled string representing a symbol name.
class SymbolStringPtr {
  friend class OrcV2CAPIHelper;
  friend class SymbolStringPool;
  friend struct DenseMapInfo<SymbolStringPtr>;

public:
  SymbolStringPtr() = default;
  SymbolStringPtr(std::nullptr_t) {}
  SymbolStringPtr(const SymbolStringPtr &Other)
    : S(Other.S) {
    if (isRealPoolEntry(S))
      ++S->getValue();
  }

  SymbolStringPtr& operator=(const SymbolStringPtr &Other) {
    if (isRealPoolEntry(S)) {
      assert(S->getValue() && "Releasing SymbolStringPtr with zero ref count");
      --S->getValue();
    }
    S = Other.S;
    if (isRealPoolEntry(S))
      ++S->getValue();
    return *this;
  }

  SymbolStringPtr(SymbolStringPtr &&Other) : S(nullptr) {
    std::swap(S, Other.S);
  }

  SymbolStringPtr& operator=(SymbolStringPtr &&Other) {
    if (isRealPoolEntry(S)) {
      assert(S->getValue() && "Releasing SymbolStringPtr with zero ref count");
      --S->getValue();
    }
    S = nullptr;
    std::swap(S, Other.S);
    return *this;
  }

  ~SymbolStringPtr() {
    if (isRealPoolEntry(S)) {
      assert(S->getValue() && "Releasing SymbolStringPtr with zero ref count");
      --S->getValue();
    }
  }

  explicit operator bool() const { return S; }

  StringRef operator*() const { return S->first(); }

  friend bool operator==(const SymbolStringPtr &LHS,
                         const SymbolStringPtr &RHS) {
    return LHS.S == RHS.S;
  }

  friend bool operator!=(const SymbolStringPtr &LHS,
                         const SymbolStringPtr &RHS) {
    return !(LHS == RHS);
  }

  friend bool operator<(const SymbolStringPtr &LHS,
                        const SymbolStringPtr &RHS) {
    return LHS.S < RHS.S;
  }

private:
  using PoolEntry = SymbolStringPool::PoolMapEntry;
  using PoolEntryPtr = PoolEntry *;

  SymbolStringPtr(SymbolStringPool::PoolMapEntry *S)
      : S(S) {
    if (isRealPoolEntry(S))
      ++S->getValue();
  }

  // Returns false for null, empty, and tombstone values, true otherwise.
  bool isRealPoolEntry(PoolEntryPtr P) {
    return ((reinterpret_cast<uintptr_t>(P) - 1) & InvalidPtrMask) !=
           InvalidPtrMask;
  }

  static SymbolStringPtr getEmptyVal() {
    return SymbolStringPtr(reinterpret_cast<PoolEntryPtr>(EmptyBitPattern));
  }

  static SymbolStringPtr getTombstoneVal() {
    return SymbolStringPtr(reinterpret_cast<PoolEntryPtr>(TombstoneBitPattern));
  }

  constexpr static uintptr_t EmptyBitPattern =
      std::numeric_limits<uintptr_t>::max()
      << PointerLikeTypeTraits<PoolEntryPtr>::NumLowBitsAvailable;

  constexpr static uintptr_t TombstoneBitPattern =
      (std::numeric_limits<uintptr_t>::max() - 1)
      << PointerLikeTypeTraits<PoolEntryPtr>::NumLowBitsAvailable;

  constexpr static uintptr_t InvalidPtrMask =
      (std::numeric_limits<uintptr_t>::max() - 3)
      << PointerLikeTypeTraits<PoolEntryPtr>::NumLowBitsAvailable;

  PoolEntryPtr S = nullptr;
};

inline SymbolStringPool::~SymbolStringPool() {
#ifndef NDEBUG
  clearDeadEntries();
  assert(Pool.empty() && "Dangling references at pool destruction time");
#endif // NDEBUG
}

inline SymbolStringPtr SymbolStringPool::intern(StringRef S) {
  std::lock_guard<std::mutex> Lock(PoolMutex);
  PoolMap::iterator I;
  bool Added;
  std::tie(I, Added) = Pool.try_emplace(S, 0);
  return SymbolStringPtr(&*I);
}

inline void SymbolStringPool::clearDeadEntries() {
  std::lock_guard<std::mutex> Lock(PoolMutex);
  for (auto I = Pool.begin(), E = Pool.end(); I != E;) {
    auto Tmp = I++;
    if (Tmp->second == 0)
      Pool.erase(Tmp);
  }
}

inline bool SymbolStringPool::empty() const {
  std::lock_guard<std::mutex> Lock(PoolMutex);
  return Pool.empty();
}

} // end namespace orc

template <>
struct DenseMapInfo<orc::SymbolStringPtr> {

  static orc::SymbolStringPtr getEmptyKey() {
    return orc::SymbolStringPtr::getEmptyVal();
  }

  static orc::SymbolStringPtr getTombstoneKey() {
    return orc::SymbolStringPtr::getTombstoneVal();
  }

  static unsigned getHashValue(const orc::SymbolStringPtr &V) {
    return DenseMapInfo<orc::SymbolStringPtr::PoolEntryPtr>::getHashValue(V.S);
  }

  static bool isEqual(const orc::SymbolStringPtr &LHS,
                      const orc::SymbolStringPtr &RHS) {
    return LHS.S == RHS.S;
  }
};

} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_SYMBOLSTRINGPOOL_H
