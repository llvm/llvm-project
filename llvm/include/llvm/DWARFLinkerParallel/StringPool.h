//===- StringPool.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DWARFLINKERPARALLEL_STRINGPOOL_H
#define LLVM_DWARFLINKERPARALLEL_STRINGPOOL_H

#include "llvm/ADT/ConcurrentHashtable.h"
#include "llvm/CodeGen/DwarfStringPoolEntry.h"
#include "llvm/Support/Allocator.h"
#include <string>
#include <string_view>

namespace llvm {
namespace dwarflinker_parallel {

/// StringEntry keeps data of the string: the length, external offset
/// and a string body which is placed right after StringEntry.
using StringEntry = StringMapEntry<DwarfStringPoolEntry *>;

class PerThreadStringAllocator
    : public AllocatorBase<PerThreadStringAllocator> {
public:
  inline LLVM_ATTRIBUTE_RETURNS_NONNULL void *Allocate(size_t Size,
                                                       size_t Alignment) {
    return ThreadLocalAllocator.Allocate(Size, Align(Alignment));
  }

  // Pull in base class overloads.
  using AllocatorBase<PerThreadStringAllocator>::Allocate;

private:
  static thread_local BumpPtrAllocator ThreadLocalAllocator;
};

class StringPoolEntryInfo {
public:
  /// \returns Hash value for the specified \p Key.
  static inline uint64_t getHashValue(const StringRef &Key) {
    return xxHash64(Key);
  }

  /// \returns true if both \p LHS and \p RHS are equal.
  static inline bool isEqual(const StringRef &LHS, const StringRef &RHS) {
    return LHS == RHS;
  }

  /// \returns key for the specified \p KeyData.
  static inline StringRef getKey(const StringEntry &KeyData) {
    return KeyData.getKey();
  }

  /// \returns newly created object of KeyDataTy type.
  static inline StringEntry *create(const StringRef &Key,
                                    PerThreadStringAllocator &Allocator) {
    return StringEntry::create(Key, Allocator);
  }
};

class StringPool : public ConcurrentHashTableByPtr<StringRef, StringEntry,
                                                   PerThreadStringAllocator,
                                                   StringPoolEntryInfo> {
public:
  StringPool()
      : ConcurrentHashTableByPtr<StringRef, StringEntry,
                                 PerThreadStringAllocator, StringPoolEntryInfo>(
            Allocator) {}

  StringPool(size_t InitialSize)
      : ConcurrentHashTableByPtr<StringRef, StringEntry,
                                 PerThreadStringAllocator, StringPoolEntryInfo>(
            Allocator, InitialSize) {}

private:
  PerThreadStringAllocator Allocator;
};

} // end of namespace dwarflinker_parallel
} // end namespace llvm

#endif // LLVM_DWARFLINKERPARALLEL_STRINGPOOL_H
