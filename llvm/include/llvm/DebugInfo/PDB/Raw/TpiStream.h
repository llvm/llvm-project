//===- TpiStream.cpp - PDB Type Info (TPI) Stream 2 Access ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_PDBTPISTREAM_H
#define LLVM_DEBUGINFO_PDB_RAW_PDBTPISTREAM_H

#include "llvm/DebugInfo/CodeView/TypeStream.h"
#include "llvm/DebugInfo/PDB/PDBTypes.h"
#include "llvm/DebugInfo/PDB/Raw/ByteStream.h"
#include "llvm/DebugInfo/PDB/Raw/MappedBlockStream.h"
#include "llvm/DebugInfo/PDB/Raw/RawConstants.h"

namespace llvm {
namespace pdb {
class PDBFile;

typedef uint32_t (*HashFunctionType)(uint8_t *, uint32_t);

class TpiStream {
  struct HeaderInfo;

public:
  TpiStream(PDBFile &File);
  ~TpiStream();
  std::error_code reload();

  PdbRaw_TpiVer getTpiVersion() const;

  uint32_t TypeIndexBegin() const;
  uint32_t TypeIndexEnd() const;
  uint32_t NumTypeRecords() const;

  iterator_range<codeview::TypeIterator> types() const;

private:
  PDBFile &Pdb;
  MappedBlockStream Stream;
  HashFunctionType HashFunction;

  ByteStream RecordsBuffer;
  ByteStream TypeIndexOffsetBuffer;
  ByteStream HashValuesBuffer;
  ByteStream HashAdjBuffer;

  std::unique_ptr<HeaderInfo> Header;
};
}
}

#endif
