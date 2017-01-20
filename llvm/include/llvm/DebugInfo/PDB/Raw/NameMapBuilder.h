//===- NameMapBuilder.h - PDB Name Map Builder ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_PDBNAMEMAPBUILDER_H
#define LLVM_DEBUGINFO_PDB_RAW_PDBNAMEMAPBUILDER_H

#include "llvm/DebugInfo/PDB/Raw/HashTable.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace llvm {
namespace msf {
class StreamWriter;
}
namespace pdb {
class NameMap;

class NameMapBuilder {
public:
  NameMapBuilder();

  void addMapping(StringRef Name, uint32_t Mapping);

  Error commit(msf::StreamWriter &Writer) const;

  uint32_t calculateSerializedLength() const;

private:
  std::vector<StringRef> Strings;
  HashTable Map;
  uint32_t Offset = 0;
};

} // end namespace pdb
} // end namespace llvm

#endif // LLVM_DEBUGINFO_PDB_RAW_PDBNAMEMAPBUILDER_H
