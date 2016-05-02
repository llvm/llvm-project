//===- NameHashTable.h - PDB Name Hash Table --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_NAMEHASHTABLE_H
#define LLVM_DEBUGINFO_PDB_RAW_NAMEHASHTABLE_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/PDB/Raw/ByteStream.h"

#include <stdint.h>
#include <utility>

namespace llvm {
namespace pdb {
class StreamReader;
class NameHashTable {
public:
  NameHashTable();

  std::error_code load(StreamReader &Stream);

  uint32_t getNameCount() const { return NameCount; }
  uint32_t getHashVersion() const { return HashVersion; }
  uint32_t getSignature() const { return Signature; }

  StringRef getStringForID(uint32_t ID) const;
  uint32_t getIDForString(StringRef Str) const;

  ArrayRef<uint32_t> name_ids() const;

private:
  ByteStream NamesBuffer;
  std::vector<uint32_t> IDs;
  uint32_t Signature;
  uint32_t HashVersion;
  uint32_t NameCount;
};
}
}

#endif
