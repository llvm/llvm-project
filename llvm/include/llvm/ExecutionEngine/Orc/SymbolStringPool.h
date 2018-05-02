//===- SymbolStringPool.h - Multi-threaded pool for JIT symbols -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Contains a multi-threaded string pool suitable for use with ORC.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_SYMBOLSTRINGPOOL_H
#define LLVM_EXECUTIONENGINE_ORC_SYMBOLSTRINGPOOL_H

#include "llvm/ADT/StringMap.h"
#include <atomic>
#include <mutex>

namespace llvm {
namespace orc {

class SymbolStringPtr;

/// String pool for symbol names used by the JIT.
class SymbolStringPool {
  friend class SymbolStringPtr;
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
  friend class SymbolStringPool;
  friend bool operator==(const SymbolStringPtr &LHS,
                         const SymbolStringPtr &RHS);
  friend bool operator<(const SymbolStringPtr &LHS, const SymbolStringPtr &RHS);

public:
  SymbolStringPtr() = default;
  SymbolStringPtr(const SymbolStringPtr &Other)
    : S(Other.S) {
    if (S)
      ++S->getValue();
  }

  SymbolStringPtr& operator=(const SymbolStringPtr &Other) {
    if (S)
      --S->getValue();
    S = Other.S;
    if (S)
      ++S->getValue();
    return *this;
  }

  SymbolStringPtr(SymbolStringPtr &&Other) : S(nullptr) {
    std::swap(S, Other.S);
  }

  SymbolStringPtr& operator=(SymbolStringPtr &&Other) {
    if (S)
      --S->getValue();
    S = nullptr;
    std::swap(S, Other.S);
    return *this;
  }

  ~SymbolStringPtr() {
    if (S)
      --S->getValue();
  }

  StringRef operator*() const { return S->first(); }

private:

  SymbolStringPtr(SymbolStringPool::PoolMapEntry *S)
      : S(S) {
    if (S)
      ++S->getValue();
  }

  SymbolStringPool::PoolMapEntry *S = nullptr;
};

inline bool operator==(const SymbolStringPtr &LHS, const SymbolStringPtr &RHS) {
  return LHS.S == RHS.S;
}

inline bool operator!=(const SymbolStringPtr &LHS, const SymbolStringPtr &RHS) {
  return !(LHS == RHS);
}

inline bool operator<(const SymbolStringPtr &LHS, const SymbolStringPtr &RHS) {
  return LHS.S < RHS.S;
}

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
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_SYMBOLSTRINGPOOL_H
