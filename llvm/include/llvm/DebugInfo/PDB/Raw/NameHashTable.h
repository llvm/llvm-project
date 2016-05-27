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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/CodeView/StreamArray.h"
#include "llvm/DebugInfo/CodeView/StreamRef.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <vector>

namespace llvm {
namespace codeview {
class StreamReader;
}
namespace pdb {

class NameHashTable {
public:
  NameHashTable();

  Error load(codeview::StreamReader &Stream);

  uint32_t getNameCount() const { return NameCount; }
  uint32_t getHashVersion() const { return HashVersion; }
  uint32_t getSignature() const { return Signature; }

  StringRef getStringForID(uint32_t ID) const;
  uint32_t getIDForString(StringRef Str) const;

  codeview::FixedStreamArray<support::ulittle32_t> name_ids() const;

private:
  codeview::StreamRef NamesBuffer;
  codeview::FixedStreamArray<support::ulittle32_t> IDs;
  uint32_t Signature;
  uint32_t HashVersion;
  uint32_t NameCount;
};

} // end namespace pdb
} // end namespace llvm

#endif // LLVM_DEBUGINFO_PDB_RAW_NAMEHASHTABLE_H
