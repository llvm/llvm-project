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

#include "llvm/DebugInfo/CodeView/StreamArray.h"
#include "llvm/DebugInfo/CodeView/StreamRef.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/PDB/PDBTypes.h"
#include "llvm/DebugInfo/PDB/Raw/MappedBlockStream.h"
#include "llvm/DebugInfo/PDB/Raw/RawConstants.h"
#include "llvm/DebugInfo/PDB/Raw/RawTypes.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Support/Error.h"

namespace llvm {
namespace pdb {
class PDBFile;

class TpiStream {
  struct HeaderInfo;

public:
  TpiStream(const PDBFile &File, std::unique_ptr<MappedBlockStream> Stream);
  ~TpiStream();
  Error reload();

  PdbRaw_TpiVer getTpiVersion() const;

  uint32_t TypeIndexBegin() const;
  uint32_t TypeIndexEnd() const;
  uint32_t NumTypeRecords() const;
  uint16_t getTypeHashStreamIndex() const;
  uint16_t getTypeHashStreamAuxIndex() const;

  uint32_t getHashKeySize() const;
  uint32_t NumHashBuckets() const;
  codeview::FixedStreamArray<support::ulittle32_t> getHashValues() const;
  codeview::FixedStreamArray<TypeIndexOffset> getTypeIndexOffsets() const;
  codeview::FixedStreamArray<TypeIndexOffset> getHashAdjustments() const;

  iterator_range<codeview::CVTypeArray::Iterator> types(bool *HadError) const;

  Error commit();

private:
  Error verifyHashValues();

  const PDBFile &Pdb;
  std::unique_ptr<MappedBlockStream> Stream;

  codeview::CVTypeArray TypeRecords;

  std::unique_ptr<MappedBlockStream> HashStream;
  codeview::FixedStreamArray<support::ulittle32_t> HashValues;
  codeview::FixedStreamArray<TypeIndexOffset> TypeIndexOffsets;
  codeview::FixedStreamArray<TypeIndexOffset> HashAdjustments;

  const HeaderInfo *Header;
};
}
}

#endif
