//===- PDBStringTable.h - PDB String Table -----------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_NATIVE_PDBSTRINGTABLE_H
#define LLVM_DEBUGINFO_PDB_NATIVE_PDBSTRINGTABLE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/CodeView/DebugStringTableSubsection.h"
#include "llvm/Support/BinaryStreamArray.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include <cstdint>

namespace llvm {
class BinaryStreamReader;

namespace pdb {

struct PDBStringTableHeader;

class PDBStringTable {
public:
  LLVM_ABI Error reload(BinaryStreamReader &Reader);

  LLVM_ABI uint32_t getByteSize() const;
  LLVM_ABI uint32_t getNameCount() const;
  LLVM_ABI uint32_t getHashVersion() const;
  LLVM_ABI uint32_t getSignature() const;

  LLVM_ABI Expected<StringRef> getStringForID(uint32_t ID) const;
  LLVM_ABI Expected<uint32_t> getIDForString(StringRef Str) const;

  LLVM_ABI FixedStreamArray<support::ulittle32_t> name_ids() const;

  LLVM_ABI const codeview::DebugStringTableSubsectionRef &
  getStringTable() const;

private:
  Error readHeader(BinaryStreamReader &Reader);
  Error readStrings(BinaryStreamReader &Reader);
  Error readHashTable(BinaryStreamReader &Reader);
  Error readEpilogue(BinaryStreamReader &Reader);

  const PDBStringTableHeader *Header = nullptr;
  codeview::DebugStringTableSubsectionRef Strings;
  FixedStreamArray<support::ulittle32_t> IDs;
  uint32_t NameCount = 0;
};

} // end namespace pdb
} // end namespace llvm

#endif // LLVM_DEBUGINFO_PDB_NATIVE_PDBSTRINGTABLE_H
