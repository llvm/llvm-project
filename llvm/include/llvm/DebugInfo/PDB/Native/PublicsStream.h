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

#include "llvm/DebugInfo/CodeView/SymbolRecord.h"
#include "llvm/DebugInfo/MSF/MappedBlockStream.h"
#include "llvm/DebugInfo/PDB/Native/RawConstants.h"
#include "llvm/DebugInfo/PDB/Native/RawTypes.h"
#include "llvm/DebugInfo/PDB/PDBTypes.h"
#include "llvm/Support/BinaryStreamArray.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace pdb {
class DbiStream;
struct GSIHashHeader;
class PDBFile;

class PublicsStream {
public:
  PublicsStream(PDBFile &File, std::unique_ptr<msf::MappedBlockStream> Stream);
  ~PublicsStream();
  Error reload();

  uint32_t getSymHash() const;
  uint32_t getAddrMap() const;
  uint32_t getNumBuckets() const { return NumBuckets; }
  Expected<const codeview::CVSymbolArray &> getSymbolArray() const;
  iterator_range<codeview::CVSymbolArray::Iterator>
  getSymbols(bool *HadError) const;
  FixedStreamArray<support::ulittle32_t> getHashBuckets() const {
    return HashBuckets;
  }
  FixedStreamArray<support::ulittle32_t> getAddressMap() const {
    return AddressMap;
  }
  FixedStreamArray<support::ulittle32_t> getThunkMap() const {
    return ThunkMap;
  }
  FixedStreamArray<SectionOffset> getSectionOffsets() const {
    return SectionOffsets;
  }

  Error commit();

private:
  PDBFile &Pdb;

  std::unique_ptr<msf::MappedBlockStream> Stream;
  uint32_t NumBuckets = 0;
  ArrayRef<uint8_t> Bitmap;
  FixedStreamArray<PSHashRecord> HashRecords;
  FixedStreamArray<support::ulittle32_t> HashBuckets;
  FixedStreamArray<support::ulittle32_t> AddressMap;
  FixedStreamArray<support::ulittle32_t> ThunkMap;
  FixedStreamArray<SectionOffset> SectionOffsets;

  const PublicsStreamHeader *Header;
  const GSIHashHeader *HashHdr;
};
}
}

#endif
