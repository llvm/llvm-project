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
class raw_ostream;

class CachedHashString {
  const char *P;
  uint32_t Size;
  uint32_t Hash;

public:
  CachedHashString(StringRef S)
      : CachedHashString(S, DenseMapInfo<StringRef>::getHashValue(S)) {}
  CachedHashString(StringRef S, uint32_t Hash)
      : P(S.data()), Size(S.size()), Hash(Hash) {
    assert(S.size() <= std::numeric_limits<uint32_t>::max());
  }

  StringRef val() const { return StringRef(P, Size); }
  uint32_t size() const { return Size; }
  uint32_t hash() const { return Hash; }
};

/// \brief Utility for building string tables with deduplicated suffixes.
class StringTableBuilder {
public:
  enum Kind { ELF, WinCOFF, MachO, RAW };

private:
  DenseMap<CachedHashString, size_t> StringIndexMap;
  size_t Size = 0;
  Kind K;
  unsigned Alignment;
  bool Finalized = false;

  void finalizeStringTable(bool Optimize);
  void initSize();

public:
  StringTableBuilder(Kind K, unsigned Alignment = 1);
  ~StringTableBuilder();

  /// \brief Add a string to the builder. Returns the position of S in the
  /// table. The position will be changed if finalize is used.
  /// Can only be used before the table is finalized.
  size_t add(CachedHashString S);
  size_t add(StringRef S) { return add(CachedHashString(S)); }

  /// \brief Analyze the strings and build the final table. No more strings can
  /// be added after this point.
  void finalize();

  /// Finalize the string table without reording it. In this mode, offsets
  /// returned by add will still be valid.
  void finalizeInOrder();

  /// \brief Get the offest of a string in the string table. Can only be used
  /// after the table is finalized.
  size_t getOffset(CachedHashString S) const;
  size_t getOffset(StringRef S) const { return getOffset(CachedHashString(S)); }

  size_t getSize() const { return Size; }
  void clear();

  void write(raw_ostream &OS) const;
  void write(uint8_t *Buf) const;

private:
  bool isFinalized() const { return Finalized; }
};

} // end llvm namespace

#endif
