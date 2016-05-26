//===- NameMap.h - PDB Name Map ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_PDBNAMEMAP_H
#define LLVM_DEBUGINFO_PDB_RAW_PDBNAMEMAP_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <cstdint>

namespace llvm {
namespace codeview {
class StreamReader;
}
namespace pdb {

class NameMap {
public:
  NameMap();

  Error load(codeview::StreamReader &Stream);

  bool tryGetValue(StringRef Name, uint32_t &Value) const;

  iterator_range<StringMapConstIterator<uint32_t>> entries() const;

private:
  StringMap<uint32_t> Mapping;
};

} // end namespace pdb
} // end namespace llvm

#endif // LLVM_DEBUGINFO_PDB_RAW_PDBNAMEMAP_H
