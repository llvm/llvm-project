//===-- StringTableBuilder.h - String table building utility ------*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_STRINGTABLEBUILDER_H
#define LLVM_MC_STRINGTABLEBUILDER_H

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/DenseMap.h"
#include <cassert>

namespace llvm {

/// \brief Utility for building string tables with deduplicated suffixes.
class StringTableBuilder {
public:
  enum Kind { ELF, WinCOFF, MachO, RAW };

private:
  SmallString<256> StringTable;
  DenseMap<CachedHash<StringRef>, size_t> StringIndexMap;
  size_t Size = 0;
  Kind K;
  unsigned Alignment;

  void finalizeStringTable(bool Optimize);

public:
  StringTableBuilder(Kind K, unsigned Alignment = 1);

  /// \brief Add a string to the builder. Returns the position of S in the
  /// table. The position will be changed if finalize is used.
  /// Can only be used before the table is finalized.
  size_t add(StringRef S);

  /// \brief Analyze the strings and build the final table. No more strings can
  /// be added after this point.
  void finalize();

  /// Finalize the string table without reording it. In this mode, offsets
  /// returned by add will still be valid.
  void finalizeInOrder();

  /// \brief Retrieve the string table data. Can only be used after the table
  /// is finalized.
  StringRef data() const {
    assert(isFinalized());
    return StringTable;
  }

  /// \brief Get the offest of a string in the string table. Can only be used
  /// after the table is finalized.
  size_t getOffset(StringRef S) const;

  const DenseMap<CachedHash<StringRef>, size_t> &getMap() const {
    return StringIndexMap;
  }

  size_t getSize() const { return Size; }
  void clear();

private:
  bool isFinalized() const {
    return !StringTable.empty();
  }
};

} // end llvm namespace

#endif
