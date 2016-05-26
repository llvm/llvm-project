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
#include "llvm/DebugInfo/CodeView/TypeStream.h"
#include "llvm/DebugInfo/PDB/PDBTypes.h"
#include "llvm/DebugInfo/PDB/Raw/MappedBlockStream.h"
#include "llvm/DebugInfo/PDB/Raw/RawConstants.h"

#include "llvm/Support/Error.h"

namespace llvm {
namespace pdb {
class DbiStream;
class PDBFile;

class PublicsStream {
  struct GSIHashHeader;
  struct HashRecord;
  struct HeaderInfo;

public:
  PublicsStream(PDBFile &File, uint32_t StreamNum);
  ~PublicsStream();
  Error reload();

  uint32_t getStreamNum() const { return StreamNum; }
  uint32_t getSymHash() const;
  uint32_t getAddrMap() const;
  uint32_t getNumBuckets() const { return NumBuckets; }
  iterator_range<codeview::SymbolIterator> getSymbols() const;
  ArrayRef<uint32_t> getHashBuckets() const { return HashBuckets; }
  ArrayRef<uint32_t> getAddressMap() const { return AddressMap; }
  ArrayRef<uint32_t> getThunkMap() const { return ThunkMap; }
  ArrayRef<uint32_t> getSectionOffsets() const { return SectionOffsets; }

private:
  Error readSymbols();

  PDBFile &Pdb;

  uint32_t StreamNum;
  MappedBlockStream Stream;
  uint32_t NumBuckets = 0;
  std::vector<HashRecord> HashRecords;
  std::vector<uint32_t> HashBuckets;
  std::vector<uint32_t> AddressMap;
  std::vector<uint32_t> ThunkMap;
  std::vector<uint32_t> SectionOffsets;

  std::unique_ptr<HeaderInfo> Header;
  std::unique_ptr<GSIHashHeader> HashHdr;
};
}
}

#endif
