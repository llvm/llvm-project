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
class PublicsStream;
class SymbolStream;
class TpiStream;

static const char MsfMagic[] = {'M',  'i',  'c',    'r', 'o', 's',  'o',  'f',
                                't',  ' ',  'C',    '/', 'C', '+',  '+',  ' ',
                                'M',  'S',  'F',    ' ', '7', '.',  '0',  '0',
                                '\r', '\n', '\x1a', 'D', 'S', '\0', '\0', '\0'};

class PDBFile : public IPDBFile {
public:
  // The superblock is overlaid at the beginning of the file (offset 0).
  // It starts with a magic header and is followed by information which
  // describes the layout of the file system.
  struct SuperBlock {
    char MagicBytes[sizeof(MsfMagic)];
    // The file system is split into a variable number of fixed size elements.
    // These elements are referred to as blocks.  The size of a block may vary
    // from system to system.
    support::ulittle32_t BlockSize;
    // This field's purpose is not yet known.
    support::ulittle32_t Unknown0;
    // This contains the number of blocks resident in the file system.  In
    // practice, NumBlocks * BlockSize is equivalent to the size of the PDB
    // file.
    support::ulittle32_t NumBlocks;
    // This contains the number of bytes which make up the directory.
    support::ulittle32_t NumDirectoryBytes;
    // This field's purpose is not yet known.
    support::ulittle32_t Unknown1;
    // This contains the block # of the block map.
    support::ulittle32_t BlockMapAddr;
  };

  explicit PDBFile(std::unique_ptr<codeview::StreamInterface> PdbFileBuffer);
  ~PDBFile() override;

  uint32_t getUnknown0() const;
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
  size_t getFileSize() const;

  ArrayRef<uint8_t> getBlockData(uint32_t BlockIndex,
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

  static uint64_t bytesToBlocks(uint64_t NumBytes, uint64_t BlockSize) {
    return alignTo(NumBytes, BlockSize) / BlockSize;
  }

  static uint64_t blockToOffset(uint64_t BlockNumber, uint64_t BlockSize) {
    return BlockNumber * BlockSize;
  }

  Expected<InfoStream &> getPDBInfoStream();
  Expected<DbiStream &> getPDBDbiStream();
  Expected<TpiStream &> getPDBTpiStream();
  Expected<TpiStream &> getPDBIpiStream();
  Expected<PublicsStream &> getPDBPublicsStream();
  Expected<SymbolStream &> getPDBSymbolStream();
  Expected<NameHashTable &> getStringTable();

  void setSuperBlock(const SuperBlock *Block);
  void setStreamSizes(ArrayRef<support::ulittle32_t> Sizes);
  void setStreamMap(ArrayRef<ArrayRef<support::ulittle32_t>> Blocks);
  void commit();

private:
  std::unique_ptr<codeview::StreamInterface> Buffer;
  const PDBFile::SuperBlock *SB;
  ArrayRef<support::ulittle32_t> StreamSizes;
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
