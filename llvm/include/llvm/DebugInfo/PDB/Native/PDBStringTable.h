//===- PDBStringTable.h - PDB String Table -------------------------*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_PDBSTRINGTABLE_H
#define LLVM_DEBUGINFO_PDB_RAW_PDBSTRINGTABLE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/BinaryStreamArray.h"
#include "llvm/Support/BinaryStreamRef.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <vector>

namespace llvm {
class BinaryStreamReader;

namespace pdb {

class PDBStringTable {
public:
  PDBStringTable();

  Error load(BinaryStreamReader &Stream);

  uint32_t getByteSize() const;

  uint32_t getNameCount() const { return NameCount; }
  uint32_t getHashVersion() const { return HashVersion; }
  uint32_t getSignature() const { return Signature; }

  StringRef getStringForID(uint32_t ID) const;
  uint32_t getIDForString(StringRef Str) const;

  FixedStreamArray<support::ulittle32_t> name_ids() const;

private:
  BinaryStreamRef NamesBuffer;
  FixedStreamArray<support::ulittle32_t> IDs;
  uint32_t ByteSize = 0;
  uint32_t Signature = 0;
  uint32_t HashVersion = 0;
  uint32_t NameCount = 0;
};

} // end namespace pdb
} // end namespace llvm

#endif // LLVM_DEBUGINFO_PDB_RAW_STRINGTABLE_H
