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
#include "llvm/DebugInfo/MSF/IMSFFile.h"
#include "llvm/DebugInfo/MSF/MSFCommon.h"
#include "llvm/DebugInfo/MSF/StreamArray.h"
#include "llvm/DebugInfo/MSF/StreamInterface.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"

#include <memory>

namespace llvm {

namespace msf {
class MappedBlockStream;
class WritableStream;
}

namespace pdb {
class DbiStream;
class InfoStream;
class NameHashTable;
class PDBFileBuilder;
class PublicsStream;
class SymbolStream;
class TpiStream;

class PDBFile : public msf::IMSFFile {
  friend PDBFileBuilder;

public:
  PDBFile(std::unique_ptr<msf::ReadableStream> PdbFileBuffer,
          BumpPtrAllocator &Allocator);
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

  ArrayRef<uint32_t> getFpmPages() const { return FpmPages; }

  ArrayRef<support::ulittle32_t> getStreamSizes() const {
    return ContainerLayout.StreamSizes;
  }
  ArrayRef<ArrayRef<support::ulittle32_t>> getStreamMap() const {
    return ContainerLayout.StreamMap;
  }

  const msf::MSFLayout &getMsfLayout() const { return ContainerLayout; }
  const msf::ReadableStream &getMsfBuffer() const { return *Buffer; }

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

  BumpPtrAllocator &getAllocator() { return Allocator; }

private:
  BumpPtrAllocator &Allocator;

  std::unique_ptr<msf::ReadableStream> Buffer;

  std::vector<uint32_t> FpmPages;
  msf::MSFLayout ContainerLayout;

  std::unique_ptr<InfoStream> Info;
  std::unique_ptr<DbiStream> Dbi;
  std::unique_ptr<TpiStream> Tpi;
  std::unique_ptr<TpiStream> Ipi;
  std::unique_ptr<PublicsStream> Publics;
  std::unique_ptr<SymbolStream> Symbols;
  std::unique_ptr<msf::MappedBlockStream> DirectoryStream;
  std::unique_ptr<msf::MappedBlockStream> StringTableStream;
  std::unique_ptr<NameHashTable> StringTable;
};
}
}

#endif
