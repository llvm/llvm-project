//===------------- StringPool.h - Interning string pool ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A thread-safe, ref-counted pool of uniqued strings.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_STRINGPOOL_H
#define ORC_RT_STRINGPOOL_H

#include <atomic>
#include <cassert>
#include <cstddef>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace orc_rt {

/// Interns strings (e.g. symbol names, paths) behind ref-counted handles. An
/// entry is kept alive as long as at least one StringPool::Ptr refers to it;
/// clearDeadEntries() reclaims entries with no owners left.
///
/// intern() and clearDeadEntries() may be called concurrently from any
/// number of threads. Copying, moving, and destroying a StringPool::Ptr
/// requires no lock -- only the atomic refcount in that ptr's own entry is
/// touched.
class StringPool {
private:
  using RefCount = std::atomic<size_t>;
  using PoolMap = std::unordered_map<std::string, RefCount>;

public:
  using PoolEntry = PoolMap::value_type;

  class PtrBase;
  class Ptr;
  class WeakPtr;
  class EntryUnsafe;

  StringPool() = default;
  StringPool(const StringPool &) = delete;
  StringPool &operator=(const StringPool &) = delete;
  ~StringPool();

  /// Returns the Ptr for S, interning a copy on first reference.
  Ptr intern(std::string_view S);

  /// Erase entries with no remaining Ptr owners.
  void clearDeadEntries();

  /// Returns true if this pool has no entries.
  bool empty() const;

private:
  mutable std::mutex M;
  PoolMap Pool;
};

/// Common base for StringPool::Ptr and StringPool::WeakPtr: bool conversion,
/// dereference, and comparison.
///
/// Comparisons and hashing are pointer-identity, scoped to whichever
/// StringPool produced the handle -- handles from different pools are never
/// equal, even for identical text.
class StringPool::PtrBase {
  friend class EntryUnsafe;

public:
  PtrBase() = default;
  PtrBase(std::nullptr_t) noexcept {}

  explicit operator bool() const noexcept { return E != nullptr; }

  const std::string &operator*() const noexcept { return E->first; }

  friend bool operator==(PtrBase LHS, PtrBase RHS) noexcept {
    return LHS.E == RHS.E;
  }
  friend bool operator!=(PtrBase LHS, PtrBase RHS) noexcept {
    return !(LHS == RHS);
  }
  // Pointer-order only; not stable across runs (ASLR). Fine as a map/set key
  // ordering, not for anything user-visible.
  friend bool operator<(PtrBase LHS, PtrBase RHS) noexcept {
    return LHS.E < RHS.E;
  }

protected:
  using PoolEntry = StringPool::PoolEntry;

  explicit PtrBase(PoolEntry *E) noexcept : E(E) {}
  PoolEntry *E = nullptr;
};

/// An owning, ref-counted handle to a string interned in some StringPool.
class StringPool::Ptr : public StringPool::PtrBase {
  friend class StringPool;

public:
  Ptr() = default;
  Ptr(std::nullptr_t) noexcept {}

  /// Constructs an owning handle from a weak one, incrementing the refcount.
  /// Other's entry must currently have a nonzero refcount (i.e. some other
  /// Ptr is known to be keeping it alive right now) -- constructing from a
  /// WeakPtr whose entry's refcount has already reached zero is undefined
  /// behavior, whether or not clearDeadEntries() has actually run yet.
  /// Reclamation can happen on any thread as soon as the count hits zero, so
  /// there is no safe window to observe "zero but not yet reclaimed".
  explicit Ptr(WeakPtr Other) noexcept;

  Ptr(const Ptr &Other) noexcept : PtrBase(Other.E) { incRef(); }

  Ptr &operator=(const Ptr &Other) noexcept {
    if (this != &Other) {
      decRef();
      E = Other.E;
      incRef();
    }
    return *this;
  }

  Ptr(Ptr &&Other) noexcept { std::swap(E, Other.E); }

  Ptr &operator=(Ptr &&Other) noexcept {
    decRef();
    E = nullptr;
    std::swap(E, Other.E);
    return *this;
  }

  ~Ptr() { decRef(); }

private:
  explicit Ptr(PoolEntry *E) noexcept : PtrBase(E) { incRef(); }

  void incRef() noexcept {
    if (E)
      ++E->second;
  }

  void decRef() noexcept {
    if (E) {
      assert(E->second.load() != 0 && "double-release of StringPool::Ptr");
      --E->second;
    }
  }
};

/// A non-owning (weak) handle to a string interned in some StringPool.
///
/// Comparable and hashable interchangeably with StringPool::Ptr (both wrap the
/// same underlying entry pointer), but copying a WeakPtr never touches the
/// refcount, so it's cheaper to pass around than a Ptr. It is invalidated the
/// instant the entry's refcount drops to zero (not when clearDeadEntries() next
/// happens to run, which may be arbitrarily later on another thread) so only
/// dereference, compare, or reconstruct a Ptr from a WeakPtr where a
/// corresponding Ptr is known to be keeping the entry alive,
/// e.g. as a lookup key into a table whose values (or a side table) hold the
/// owning Ptr for that same entry.
class StringPool::WeakPtr : public StringPool::PtrBase {
public:
  WeakPtr() = default;
  WeakPtr(std::nullptr_t) noexcept {}
  explicit WeakPtr(const Ptr &Other) noexcept : PtrBase(Other) {}
};

/// Provides unsafe (refcount-bypassing) access to the pool-entry pointer
/// underlying a StringPool::PtrBase. Used to implement std::hash, and
/// intended to grow C API support (retain/release/take-ownership operations
/// on an opaque pool-entry token) as that need arises.
class StringPool::EntryUnsafe {
public:
  using PoolEntry = StringPool::PoolEntry;

  /// Extracts the pool-entry pointer from S without affecting its refcount.
  static EntryUnsafe from(const PtrBase &S) { return EntryUnsafe(S.E); }

  const void *rawPtr() const { return E; }

private:
  EntryUnsafe(PoolEntry *E) : E(E) {}
  PoolEntry *E = nullptr;
};

inline StringPool::Ptr::Ptr(StringPool::WeakPtr Other) noexcept
    : PtrBase(Other) {
  incRef();
}

inline StringPool::~StringPool() {
#ifndef NDEBUG
  clearDeadEntries();
  assert(Pool.empty() && "Dangling StringPool::Ptr at StringPool destruction");
#endif
}

inline StringPool::Ptr StringPool::intern(std::string_view S) {
  std::scoped_lock<std::mutex> Lock(M);
  auto [I, Added] = Pool.try_emplace(std::string(S), 0);
  return Ptr(&*I);
}

inline void StringPool::clearDeadEntries() {
  std::scoped_lock<std::mutex> Lock(M);
  for (auto I = Pool.begin(), E = Pool.end(); I != E;)
    if (I->second.load() == 0)
      I = Pool.erase(I);
    else
      ++I;
}

inline bool StringPool::empty() const {
  std::scoped_lock<std::mutex> Lock(M);
  return Pool.empty();
}

} // namespace orc_rt

namespace std {
template <> struct hash<orc_rt::StringPool::Ptr> {
  size_t operator()(const orc_rt::StringPool::Ptr &S) const noexcept {
    return hash<const void *>()(
        orc_rt::StringPool::EntryUnsafe::from(S).rawPtr());
  }
};

template <> struct hash<orc_rt::StringPool::WeakPtr> {
  size_t operator()(const orc_rt::StringPool::WeakPtr &S) const noexcept {
    return hash<const void *>()(
        orc_rt::StringPool::EntryUnsafe::from(S).rawPtr());
  }
};
} // namespace std

#endif // ORC_RT_STRINGPOOL_H
