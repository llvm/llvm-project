//===- NamedStreamMap.h - PDB Named Stream Map ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_PDBNAMEDSTREAMMAP_H
#define LLVM_DEBUGINFO_PDB_RAW_PDBNAMEDSTREAMMAP_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/PDB/Native/HashTable.h"
#include "llvm/Support/Error.h"
#include <cstdint>

namespace llvm {
namespace msf {
class StreamReader;
class StreamWriter;
}
namespace pdb {
class NamedStreamMapBuilder;
class NamedStreamMap {
  struct FinalizationInfo {
    uint32_t StringDataBytes = 0;
    uint32_t SerializedLength = 0;
  };
  friend NamedStreamMapBuilder;

public:
  NamedStreamMap();

  Error load(msf::StreamReader &Stream);
  Error commit(msf::StreamWriter &Writer) const;
  uint32_t finalize();

  bool get(StringRef Stream, uint32_t &StreamNo) const;
  void set(StringRef Stream, uint32_t StreamNo);
  void remove(StringRef Stream);

  iterator_range<StringMapConstIterator<uint32_t>> entries() const;

private:
  Optional<FinalizationInfo> FinalizedInfo;
  HashTable FinalizedHashTable;
  StringMap<uint32_t> Mapping;
};

} // end namespace pdb
} // end namespace llvm

#endif // LLVM_DEBUGINFO_PDB_RAW_PDBNAMEDSTREAMMAP_H
