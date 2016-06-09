//===- PublicsStream.h - PDB Public Symbol Stream -------- ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_PUBLICSSTREAM_H
#define LLVM_DEBUGINFO_PDB_RAW_PUBLICSSTREAM_H

#include "llvm/DebugInfo/CodeView/StreamArray.h"
#include "llvm/DebugInfo/CodeView/SymbolRecord.h"
#include "llvm/DebugInfo/PDB/PDBTypes.h"
#include "llvm/DebugInfo/PDB/Raw/MappedBlockStream.h"
#include "llvm/DebugInfo/PDB/Raw/RawConstants.h"
#include "llvm/DebugInfo/PDB/Raw/RawTypes.h"

#include "llvm/Support/Error.h"

namespace llvm {
namespace pdb {
class DbiStream;
class PDBFile;

class PublicsStream {
  struct GSIHashHeader;
  struct HeaderInfo;

public:
  PublicsStream(PDBFile &File, std::unique_ptr<MappedBlockStream> Stream);
  ~PublicsStream();
  Error reload();

  uint32_t getSymHash() const;
  uint32_t getAddrMap() const;
  uint32_t getNumBuckets() const { return NumBuckets; }
  iterator_range<codeview::CVSymbolArray::Iterator>
  getSymbols(bool *HadError) const;
  codeview::FixedStreamArray<support::ulittle32_t> getHashBuckets() const {
    return HashBuckets;
  }
  codeview::FixedStreamArray<support::ulittle32_t> getAddressMap() const {
    return AddressMap;
  }
  codeview::FixedStreamArray<support::ulittle32_t> getThunkMap() const {
    return ThunkMap;
  }
  codeview::FixedStreamArray<SectionOffset> getSectionOffsets() const {
    return SectionOffsets;
  }

private:
  PDBFile &Pdb;

  std::unique_ptr<MappedBlockStream> Stream;
  uint32_t NumBuckets = 0;
  ArrayRef<uint8_t> Bitmap;
  codeview::FixedStreamArray<PSHashRecord> HashRecords;
  codeview::FixedStreamArray<support::ulittle32_t> HashBuckets;
  codeview::FixedStreamArray<support::ulittle32_t> AddressMap;
  codeview::FixedStreamArray<support::ulittle32_t> ThunkMap;
  codeview::FixedStreamArray<SectionOffset> SectionOffsets;

  const HeaderInfo *Header;
  const GSIHashHeader *HashHdr;
};
}
}

#endif
