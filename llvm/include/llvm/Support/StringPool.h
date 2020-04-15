//===- StringPool.h - Intern'd string pool ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares an interned string pool with separately malloc and
// reference counted entries.  This can reduce the cost of strings by using the
// same storage for identical strings.
//
// To intern a string:
//
//   StringPool Pool;
//   PooledStringPtr Str = Pool.intern("wakka wakka");
//
// To use the value of an interned string, use operator bool and operator*:
//
//   if (Str)
//     cerr << "the string is" << *Str << "\n";
//
// Pooled strings are immutable, but you can change a PooledStringPtr to point
// to another instance. So that interned strings can eventually be freed,
// strings in the string pool are reference-counted (automatically).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_STRINGPOOL_H
#define LLVM_SUPPORT_STRINGPOOL_H

#include "llvm/ADT/StringMap.h"

namespace llvm {

class PooledStringPtr;

/// StringPool - An interned string pool. Use the intern method to add a
/// string. Strings are removed automatically as PooledStringPtrs are
/// destroyed.
class StringPool {
  /// PooledString - This is the value of an entry in the pool's interning
  /// table.
  struct PooledString {
    StringPool *pool = nullptr; ///< So the string can remove itself.
    unsigned refcount = 0;      ///< Number of referencing PooledStringPtrs.

  public:
    PooledString() = default;
  };

  friend class PooledStringPtr;
  using Entry = StringMapEntry<PooledString>;
  StringMap<PooledString> internTable;

public:
  StringPool();
  ~StringPool();

  /// intern - Adds a string to the pool and returns a reference-counted
  /// pointer to it. No additional memory is allocated if the string already
  /// exists in the pool.
  PooledStringPtr intern(StringRef string);

  /// empty - Checks whether the pool is empty. Returns true if so.
  bool empty() const { return internTable.empty(); }
};

/// PooledStringPtr - A pointer to an interned string. Use operator bool to
/// test whether the pointer is valid, and operator * to get the string if so.
/// This is a lightweight value class with storage requirements equivalent to
/// a single pointer, but it does have reference-counting overhead when
/// copied.
class PooledStringPtr {
  using Entry = StringPool::Entry;
  Entry *entry = nullptr;

public:
  PooledStringPtr() = default;

  explicit PooledStringPtr(Entry *e) : entry(e) {
    if (entry)
      ++entry->getValue().refcount;
  }

  PooledStringPtr(const PooledStringPtr &that) : entry(that.entry) {
    if (entry)
      ++entry->getValue().refcount;
  }

  PooledStringPtr &operator=(const PooledStringPtr &that) {
    if (entry != that.entry) {
      clear();
      entry = that.entry;
      if (entry)
        ++entry->getValue().refcount;
    }
    return *this;
  }

  void clear() {
    if (!entry)
      return;
    if (--entry->getValue().refcount == 0) {
      entry->getValue().pool->internTable.remove(entry);
      MallocAllocator allocator;
      entry->Destroy(allocator);
    }
    entry = nullptr;
  }

  ~PooledStringPtr() { clear(); }

  const char *begin() const {
    assert(*this && "Attempt to dereference empty PooledStringPtr!");
    return entry->getKeyData();
  }

  const char *end() const {
    assert(*this && "Attempt to dereference empty PooledStringPtr!");
    return entry->getKeyData() + entry->getKeyLength();
  }

  unsigned size() const {
    assert(*this && "Attempt to dereference empty PooledStringPtr!");
    return entry->getKeyLength();
  }

  const char *operator*() const { return begin(); }
  explicit operator bool() const { return entry != nullptr; }

  bool operator==(const PooledStringPtr &that) const {
    return entry == that.entry;
  }
  bool operator!=(const PooledStringPtr &that) const {
    return entry != that.entry;
  }
};

} // end namespace llvm

#endif // LLVM_SUPPORT_STRINGPOOL_H
