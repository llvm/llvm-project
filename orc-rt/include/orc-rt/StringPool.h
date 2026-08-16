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

class PooledStringPtr;
class NonOwningPooledStringPtr;

/// Interns strings (e.g. symbol names, paths) behind ref-counted handles. An
/// entry is kept alive as long as at least one PooledStringPtr refers to it;
/// clearDeadEntries() reclaims entries with no owners left.
///
/// intern() and clearDeadEntries() may be called concurrently from any
/// number of threads. Copying, moving, and destroying a PooledStringPtr
/// requires no lock -- only the atomic refcount in that ptr's own entry is
/// touched.
class StringPool {
private:
  using RefCount = std::atomic<size_t>;
  using PoolMap = std::unordered_map<std::string, RefCount>;

public:
  using PoolEntry = PoolMap::value_type;

  StringPool() = default;
  StringPool(const StringPool &) = delete;
  StringPool &operator=(const StringPool &) = delete;
  ~StringPool();

  /// Returns the PooledStringPtr for S, interning a copy on first reference.
  PooledStringPtr intern(std::string_view S);

  /// Erase entries with no remaining PooledStringPtr owners.
  void clearDeadEntries();

  /// Returns true if this pool has no entries.
  bool empty() const;

private:
  mutable std::mutex M;
  PoolMap Pool;
};

/// Common base for PooledStringPtr and NonOwningPooledStringPtr: bool
/// conversion, dereference, and comparison.
///
/// Comparisons and hashing are pointer-identity, scoped to whichever
/// StringPool produced the handle -- handles from different pools are never
/// equal, even for identical text.
class PooledStringPtrBase {
  friend class StringPoolEntryUnsafe;

public:
  PooledStringPtrBase() = default;
  PooledStringPtrBase(std::nullptr_t) noexcept {}

  explicit operator bool() const noexcept { return E != nullptr; }

  const std::string &operator*() const noexcept { return E->first; }

  friend bool operator==(PooledStringPtrBase LHS,
                         PooledStringPtrBase RHS) noexcept {
    return LHS.E == RHS.E;
  }
  friend bool operator!=(PooledStringPtrBase LHS,
                         PooledStringPtrBase RHS) noexcept {
    return !(LHS == RHS);
  }
  // Pointer-order only; not stable across runs (ASLR). Fine as a map/set key
  // ordering, not for anything user-visible.
  friend bool operator<(PooledStringPtrBase LHS,
                        PooledStringPtrBase RHS) noexcept {
    return LHS.E < RHS.E;
  }

protected:
  using PoolEntry = StringPool::PoolEntry;

  explicit PooledStringPtrBase(PoolEntry *E) noexcept : E(E) {}
  PoolEntry *E = nullptr;
};

/// An owning, ref-counted handle to a string interned in some StringPool.
class PooledStringPtr : public PooledStringPtrBase {
  friend class StringPool;

public:
  PooledStringPtr() = default;
  PooledStringPtr(std::nullptr_t) noexcept {}

  /// Constructs an owning handle from a non-owning one, incrementing the
  /// refcount. Other must be backed by an entry that some PooledStringPtr is
  /// already keeping alive -- constructing from a NonOwningPooledStringPtr
  /// whose entry has already been reclaimed by clearDeadEntries() is
  /// undefined behavior.
  explicit PooledStringPtr(NonOwningPooledStringPtr Other) noexcept;

  PooledStringPtr(const PooledStringPtr &Other) noexcept
      : PooledStringPtrBase(Other.E) {
    incRef();
  }

  PooledStringPtr &operator=(const PooledStringPtr &Other) noexcept {
    if (this != &Other) {
      decRef();
      E = Other.E;
      incRef();
    }
    return *this;
  }

  PooledStringPtr(PooledStringPtr &&Other) noexcept { std::swap(E, Other.E); }

  PooledStringPtr &operator=(PooledStringPtr &&Other) noexcept {
    decRef();
    E = nullptr;
    std::swap(E, Other.E);
    return *this;
  }

  ~PooledStringPtr() { decRef(); }

private:
  explicit PooledStringPtr(PoolEntry *E) noexcept : PooledStringPtrBase(E) {
    incRef();
  }

  void incRef() noexcept {
    if (E)
      ++E->second;
  }

  void decRef() noexcept {
    if (E) {
      assert(E->second.load() != 0 && "double-release of PooledStringPtr");
      --E->second;
    }
  }
};

/// A non-owning handle to a string interned in some StringPool.
///
/// Comparable and hashable interchangeably with PooledStringPtr (both wrap the
/// same underlying entry pointer), but copying a NonOwningPooledStringPtr never
/// touches the refcount, so it's cheaper to pass around than a PooledStringPtr.
/// It is silently invalidated if the entry's refcount drops to zero and is
/// reclaimed by clearDeadEntries(), so only use it where a corresponding
/// PooledStringPtr is known to be keeping the entry alive -- e.g. as a lookup
/// key into a table whose values (or a side table) hold the owning
/// PooledStringPtr for that same entry.
class NonOwningPooledStringPtr : public PooledStringPtrBase {
public:
  NonOwningPooledStringPtr() = default;
  NonOwningPooledStringPtr(std::nullptr_t) noexcept {}
  explicit NonOwningPooledStringPtr(const PooledStringPtr &Other) noexcept
      : PooledStringPtrBase(Other) {}
};

/// Provides unsafe (refcount-bypassing) access to the pool-entry pointer
/// underlying a PooledStringPtrBase. Used to implement std::hash and C API
/// operations. Not intended for general use.
class StringPoolEntryUnsafe {
public:
  using PoolEntry = StringPool::PoolEntry;

  /// Extracts the pool-entry pointer from S without affecting its refcount.
  static StringPoolEntryUnsafe from(const PooledStringPtrBase &S) {
    return StringPoolEntryUnsafe(S.E);
  }

  const void *rawPtr() const { return E; }

private:
  StringPoolEntryUnsafe(PoolEntry *E) : E(E) {}
  PoolEntry *E = nullptr;
};

inline PooledStringPtr::PooledStringPtr(NonOwningPooledStringPtr Other) noexcept
    : PooledStringPtrBase(Other) {
  incRef();
}

inline StringPool::~StringPool() {
#ifndef NDEBUG
  clearDeadEntries();
  assert(Pool.empty() && "Dangling PooledStringPtr at StringPool destruction");
#endif
}

inline PooledStringPtr StringPool::intern(std::string_view S) {
  std::scoped_lock<std::mutex> Lock(M);
  auto [I, Added] = Pool.try_emplace(std::string(S), 0);
  return PooledStringPtr(&*I);
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
template <> struct hash<orc_rt::PooledStringPtr> {
  size_t operator()(const orc_rt::PooledStringPtr &S) const noexcept {
    return hash<const void *>()(
        orc_rt::StringPoolEntryUnsafe::from(S).rawPtr());
  }
};

template <> struct hash<orc_rt::NonOwningPooledStringPtr> {
  size_t operator()(const orc_rt::NonOwningPooledStringPtr &S) const noexcept {
    return hash<const void *>()(
        orc_rt::StringPoolEntryUnsafe::from(S).rawPtr());
  }
};
} // namespace std

#endif // ORC_RT_STRINGPOOL_H
