//===- StringTableBuilder.h - String table building utility -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_STRINGTABLEBUILDER_H
#define LLVM_MC_STRINGTABLEBUILDER_H

#include "llvm/ADT/CachedHashString.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Compiler.h"
#include <cstddef>
#include <cstdint>

namespace llvm {

class raw_ostream;

/// Utility for building string tables with deduplicated suffixes.
class StringTableBuilder {
public:
  enum Kind {
    ELF,
    WinCOFF,
    MachO,
    MachO64,
    MachOLinked,
    MachO64Linked,
    RAW,
    DWARF,
    XCOFF,
    DXContainer
  };

private:
  // Only non-zero priority will be recorded.
  DenseMap<CachedHashStringRef, uint8_t> StringPriorityMap;
  DenseMap<CachedHashStringRef, size_t> StringIndexMap;
  size_t Size = 0;
  Kind K;
  Align Alignment;
  bool Finalized = false;

  void finalizeStringTable(bool Optimize);
  void initSize();

public:
  LLVM_ABI StringTableBuilder(Kind K, Align Alignment = Align(1));
  LLVM_ABI ~StringTableBuilder();

  /// Add a string to the builder. Returns the position of S in the table. The
  /// position will be changed if finalize is used. Can only be used before the
  /// table is finalized. Priority is only useful with reordering. Strings with
  /// the same priority will be put together. Strings with higher priority are
  /// placed closer to the begin of string table. When adding same string with
  /// different priority, the maximum priority win.
  LLVM_ABI size_t add(CachedHashStringRef S, uint8_t Priority = 0);
  size_t add(StringRef S, uint8_t Priority = 0) {
    return add(CachedHashStringRef(S), Priority);
  }

  /// Analyze the strings and build the final table. No more strings can
  /// be added after this point.
  LLVM_ABI void finalize();

  /// Finalize the string table without reording it. In this mode, offsets
  /// returned by add will still be valid.
  LLVM_ABI void finalizeInOrder();

  /// Get the offest of a string in the string table. Can only be used
  /// after the table is finalized.
  LLVM_ABI size_t getOffset(CachedHashStringRef S) const;
  size_t getOffset(StringRef S) const {
    return getOffset(CachedHashStringRef(S));
  }

  /// Check if a string is contained in the string table. Since this class
  /// doesn't store the string values, this function can be used to check if
  /// storage needs to be done prior to adding the string.
  bool contains(StringRef S) const { return contains(CachedHashStringRef(S)); }
  bool contains(CachedHashStringRef S) const { return StringIndexMap.count(S); }

  bool empty() const { return StringIndexMap.empty(); }
  size_t getSize() const { return Size; }
  LLVM_ABI void clear();

  LLVM_ABI void write(raw_ostream &OS) const;
  LLVM_ABI void write(uint8_t *Buf) const;

  bool isFinalized() const { return Finalized; }
};

} // end namespace llvm

#endif // LLVM_MC_STRINGTABLEBUILDER_H
