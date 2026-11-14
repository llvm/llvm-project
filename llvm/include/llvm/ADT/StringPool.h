//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a concurrent string pool: a thread-safe, deduplicating
// owner of string data backed by a ConcurrentHashTable and a
// PerThreadBumpPtrAllocator.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_STRINGPOOL_H
#define LLVM_ADT_STRINGPOOL_H

#include "llvm/ADT/ConcurrentHashtable.h"
#include "llvm/ADT/StringMapEntry.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/PerThreadBumpPtrAllocator.h"

namespace llvm {

/// StringEntry keeps data of the string: the length, external offset and a
/// string body which is placed right after StringEntry.
using StringEntry = StringMapEntry<EmptyStringSetTag>;

/// A concurrent, deduplicating string pool.
class ConcurrentStringPool
    : public ConcurrentHashTableByPtr<
          StringRef, StringEntry, llvm::parallel::PerThreadBumpPtrAllocator> {
  using BaseTy =
      ConcurrentHashTableByPtr<StringRef, StringEntry,
                               llvm::parallel::PerThreadBumpPtrAllocator>;

public:
  ConcurrentStringPool() : BaseTy(Allocator) {}
  ConcurrentStringPool(size_t InitialSize) : BaseTy(Allocator, InitialSize) {}

  /// Intern \p S and return a stable StringRef owned by the pool.
  StringRef save(StringRef S) { return insert(S).first->getKey(); }

  llvm::parallel::PerThreadBumpPtrAllocator &getAllocatorRef() {
    return Allocator;
  }

  void clear() { Allocator.Reset(); }

private:
  llvm::parallel::PerThreadBumpPtrAllocator Allocator;
};

/// An alias for typical usage. Essentially a concurrent UniqueStringSaver.
using StringPool = ConcurrentStringPool;

} // end namespace llvm

#endif // LLVM_ADT_STRINGPOOL_H
