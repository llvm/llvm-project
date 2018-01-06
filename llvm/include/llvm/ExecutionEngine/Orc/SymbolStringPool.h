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

/// @brief String pool for symbol names used by the JIT.
class SymbolStringPool {
  friend class SymbolStringPtr;
public:
  /// @brief Create a symbol string pointer from the given string.
  SymbolStringPtr intern(StringRef S);

  /// @brief Remove from the pool any entries that are no longer referenced.
  void clearDeadEntries();

  /// @brief Returns true if the pool is empty.
  bool empty() const;
private:
  using RefCountType = std::atomic<size_t>;
  using PoolMap = StringMap<RefCountType>;
  using PoolMapEntry = StringMapEntry<RefCountType>;
  mutable std::mutex PoolMutex;
  PoolMap Pool;
};

/// @brief Pointer to a pooled string representing a symbol name.
class SymbolStringPtr {
  friend class SymbolStringPool;
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

  bool operator==(const SymbolStringPtr &Other) const {
    return S == Other.S;
  }

  bool operator!=(const SymbolStringPtr &Other) const {
    return !(*this == Other);
  }

  bool operator<(const SymbolStringPtr &Other) const {
    return S < Other.S;
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

inline SymbolStringPtr SymbolStringPool::intern(StringRef S) {
  std::lock_guard<std::mutex> Lock(PoolMutex);
  auto I = Pool.find(S);
  if (I != Pool.end())
    return SymbolStringPtr(&*I);

  bool Added;
  std::tie(I, Added) = Pool.try_emplace(S, 0);
  assert(Added && "Insert should always succeed here");
  return SymbolStringPtr(&*I);
}

inline void SymbolStringPool::clearDeadEntries() {
  std::lock_guard<std::mutex> Lock(PoolMutex);
  for (auto I = Pool.begin(), E = Pool.end(); I != E;) {
    auto Tmp = std::next(I);
    if (I->second == 0)
      Pool.erase(I);
    I = Tmp;
  }
}

inline bool SymbolStringPool::empty() const {
  std::lock_guard<std::mutex> Lock(PoolMutex);
  return Pool.empty();
}

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_SYMBOLSTRINGPOOL_H
