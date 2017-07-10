//===- NamedStreamMap.h - PDB Named Stream Map ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_NATIVE_NAMEDSTREAMMAP_H
#define LLVM_DEBUGINFO_PDB_NATIVE_NAMEDSTREAMMAP_H

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/DebugInfo/PDB/Native/HashTable.h"
#include "llvm/Support/Error.h"
#include <cstdint>

namespace llvm {

class BinaryStreamReader;
class BinaryStreamWriter;

namespace pdb {

class NamedStreamMap {
  friend class NamedStreamMapBuilder;

  struct FinalizationInfo {
    uint32_t StringDataBytes = 0;
    uint32_t SerializedLength = 0;
  };

public:
  NamedStreamMap();

  Error load(BinaryStreamReader &Stream);
  Error commit(BinaryStreamWriter &Writer) const;
  uint32_t finalize();

  uint32_t size() const;
  bool get(StringRef Stream, uint32_t &StreamNo) const;
  void set(StringRef Stream, uint32_t StreamNo);
  void remove(StringRef Stream);
  const StringMap<uint32_t> &getStringMap() const { return Mapping; }
  iterator_range<StringMapConstIterator<uint32_t>> entries() const;

private:
  Optional<FinalizationInfo> FinalizedInfo;
  HashTable FinalizedHashTable;
  StringMap<uint32_t> Mapping;
};

} // end namespace pdb

} // end namespace llvm

#endif // LLVM_DEBUGINFO_PDB_NATIVE_NAMEDSTREAMMAP_H
