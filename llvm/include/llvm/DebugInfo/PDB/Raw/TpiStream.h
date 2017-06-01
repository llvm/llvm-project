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

#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/MSF/StreamArray.h"
#include "llvm/DebugInfo/PDB/PDBTypes.h"
#include "llvm/DebugInfo/PDB/Raw/RawConstants.h"
#include "llvm/DebugInfo/PDB/Raw/RawTypes.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Support/Error.h"

namespace llvm {
namespace msf {
class MappedBlockStream;
}
namespace pdb {
class PDBFile;

class TpiStream {
  friend class TpiStreamBuilder;

public:
  TpiStream(const PDBFile &File,
            std::unique_ptr<msf::MappedBlockStream> Stream);
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
  msf::FixedStreamArray<support::ulittle32_t> getHashValues() const;
  msf::FixedStreamArray<TypeIndexOffset> getTypeIndexOffsets() const;
  msf::FixedStreamArray<TypeIndexOffset> getHashAdjustments() const;

  iterator_range<codeview::CVTypeArray::Iterator> types(bool *HadError) const;

  Error commit();

private:
  Error verifyHashValues();

  const PDBFile &Pdb;
  std::unique_ptr<msf::MappedBlockStream> Stream;

  codeview::CVTypeArray TypeRecords;

  std::unique_ptr<msf::ReadableStream> HashStream;
  msf::FixedStreamArray<support::ulittle32_t> HashValues;
  msf::FixedStreamArray<TypeIndexOffset> TypeIndexOffsets;
  msf::FixedStreamArray<TypeIndexOffset> HashAdjustments;

  const TpiStreamHeader *Header;
};
}
}

#endif
