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

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>

namespace llvm {
namespace pdb {
class NameMap;

class NameMapBuilder {
public:
  NameMapBuilder();

  void addMapping(StringRef Name, uint32_t Mapping);

  Expected<std::unique_ptr<NameMap>> build();

  uint32_t calculateSerializedLength() const;

private:
  StringMap<uint32_t> Map;
  uint32_t StringDataBytes = 0;
};

} // end namespace pdb
} // end namespace llvm

#endif // LLVM_DEBUGINFO_PDB_RAW_PDBNAMEMAPBUILDER_H
