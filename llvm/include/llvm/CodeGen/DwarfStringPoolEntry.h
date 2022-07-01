//===- llvm/CodeGen/DwarfStringPoolEntry.h - String pool entry --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_DWARFSTRINGPOOLENTRY_H
#define LLVM_CODEGEN_DWARFSTRINGPOOLENTRY_H

#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/StringMap.h"

namespace llvm {

class MCSymbol;

/// Data for a string pool entry.
struct DwarfStringPoolEntry {
  static constexpr unsigned NotIndexed = -1;

  MCSymbol *Symbol;
  uint64_t Offset;
  unsigned Index;

  bool isIndexed() const { return Index != NotIndexed; }
};

/// String pool entry reference.
class DwarfStringPoolEntryRef {
  const StringMapEntry<DwarfStringPoolEntry> *MapEntry = nullptr;

  const StringMapEntry<DwarfStringPoolEntry> *getMapEntry() const {
    return MapEntry;
  }

public:
  DwarfStringPoolEntryRef() = default;
  DwarfStringPoolEntryRef(const StringMapEntry<DwarfStringPoolEntry> &Entry)
      : MapEntry(&Entry) {}

  explicit operator bool() const { return getMapEntry(); }
  MCSymbol *getSymbol() const {
    assert(getMapEntry()->second.Symbol && "No symbol available!");
    return getMapEntry()->second.Symbol;
  }
  uint64_t getOffset() const { return getMapEntry()->second.Offset; }
  unsigned getIndex() const {
    assert(getMapEntry()->getValue().isIndexed());
    return getMapEntry()->second.Index;
  }
  StringRef getString() const { return getMapEntry()->first(); }
  /// Return the entire string pool entry for convenience.
  DwarfStringPoolEntry getEntry() const { return getMapEntry()->getValue(); }

  bool operator==(const DwarfStringPoolEntryRef &X) const {
    return getMapEntry() == X.getMapEntry();
  }
  bool operator!=(const DwarfStringPoolEntryRef &X) const {
    return getMapEntry() != X.getMapEntry();
  }
};

} // end namespace llvm

#endif
