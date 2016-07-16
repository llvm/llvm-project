//===- PDBFile.h - Low level interface to a PDB file ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_PDBFILE_H
#define LLVM_DEBUGINFO_PDB_RAW_PDBFILE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/DebugInfo/CodeView/StreamArray.h"
#include "llvm/DebugInfo/CodeView/StreamInterface.h"
#include "llvm/DebugInfo/PDB/Raw/IPDBFile.h"
#include "llvm/DebugInfo/PDB/Raw/MsfCommon.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"

#include <memory>

namespace llvm {

namespace codeview {
class StreamInterface;
}

namespace pdb {
class DbiStream;
class InfoStream;
class MappedBlockStream;
class NameHashTable;
class PDBFileBuilder;
class PublicsStream;
class SymbolStream;
class TpiStream;

class PDBFile : public IPDBFile {
  friend PDBFileBuilder;

public:
  explicit PDBFile(std::unique_ptr<codeview::StreamInterface> PdbFileBuffer);
  ~PDBFile() override;

  uint32_t getFreeBlockMapBlock() const;
  uint32_t getUnknown1() const;

  uint32_t getBlockSize() const override;
  uint32_t getBlockCount() const override;
  uint32_t getNumDirectoryBytes() const;
  uint32_t getBlockMapIndex() const;
  uint32_t getNumDirectoryBlocks() const;
  uint64_t getBlockMapOffset() const;

  uint32_t getNumStreams() const override;
  uint32_t getStreamByteSize(uint32_t StreamIndex) const override;
  ArrayRef<support::ulittle32_t>
  getStreamBlockList(uint32_t StreamIndex) const override;
  uint32_t getFileSize() const;

  Expected<ArrayRef<uint8_t>> getBlockData(uint32_t BlockIndex,
                                           uint32_t NumBytes) const override;
  Error setBlockData(uint32_t BlockIndex, uint32_t Offset,
                     ArrayRef<uint8_t> Data) const override;

  ArrayRef<support::ulittle32_t> getStreamSizes() const { return StreamSizes; }
  ArrayRef<ArrayRef<support::ulittle32_t>> getStreamMap() const {
    return StreamMap;
  }

  ArrayRef<support::ulittle32_t> getDirectoryBlockArray() const;

  Error parseFileHeaders();
  Error parseStreamData();

  Expected<InfoStream &> getPDBInfoStream();
  Expected<DbiStream &> getPDBDbiStream();
  Expected<TpiStream &> getPDBTpiStream();
  Expected<TpiStream &> getPDBIpiStream();
  Expected<PublicsStream &> getPDBPublicsStream();
  Expected<SymbolStream &> getPDBSymbolStream();
  Expected<NameHashTable &> getStringTable();

  Error commit();

private:
  Error setSuperBlock(const msf::SuperBlock *Block);

  BumpPtrAllocator Allocator;

  std::unique_ptr<codeview::StreamInterface> Buffer;
  const msf::SuperBlock *SB;
  ArrayRef<support::ulittle32_t> StreamSizes;
  ArrayRef<support::ulittle32_t> DirectoryBlocks;
  std::vector<ArrayRef<support::ulittle32_t>> StreamMap;

  std::unique_ptr<InfoStream> Info;
  std::unique_ptr<DbiStream> Dbi;
  std::unique_ptr<TpiStream> Tpi;
  std::unique_ptr<TpiStream> Ipi;
  std::unique_ptr<PublicsStream> Publics;
  std::unique_ptr<SymbolStream> Symbols;
  std::unique_ptr<MappedBlockStream> DirectoryStream;
  std::unique_ptr<MappedBlockStream> StringTableStream;
  std::unique_ptr<NameHashTable> StringTable;
};
}
}

#endif
