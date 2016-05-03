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
#include "llvm/Support/Endian.h"
#include "llvm/Support/MathExtras.h"

#include <memory>

namespace llvm {
class MemoryBuffer;

namespace pdb {
struct PDBFileContext;
class DbiStream;
class InfoStream;
class TpiStream;

class PDBFile {
public:
  explicit PDBFile(std::unique_ptr<MemoryBuffer> MemBuffer);
  ~PDBFile();

  uint32_t getBlockSize() const;
  uint32_t getUnknown0() const;
  uint32_t getBlockCount() const;
  uint32_t getNumDirectoryBytes() const;
  uint32_t getBlockMapIndex() const;
  uint32_t getUnknown1() const;
  uint32_t getNumDirectoryBlocks() const;
  uint64_t getBlockMapOffset() const;

  uint32_t getNumStreams() const;
  uint32_t getStreamByteSize(uint32_t StreamIndex) const;
  ArrayRef<uint32_t> getStreamBlockList(uint32_t StreamIndex) const;

  StringRef getBlockData(uint32_t BlockIndex, uint32_t NumBytes) const;

  ArrayRef<support::ulittle32_t> getDirectoryBlockArray();

  std::error_code parseFileHeaders();
  std::error_code parseStreamData();

  static uint64_t bytesToBlocks(uint64_t NumBytes, uint64_t BlockSize) {
    return alignTo(NumBytes, BlockSize) / BlockSize;
  }

  static uint64_t blockToOffset(uint64_t BlockNumber, uint64_t BlockSize) {
    return BlockNumber * BlockSize;
  }

  InfoStream &getPDBInfoStream();
  DbiStream &getPDBDbiStream();
  TpiStream &getPDBTpiStream();

private:
  std::unique_ptr<PDBFileContext> Context;
  std::unique_ptr<InfoStream> Info;
  std::unique_ptr<DbiStream> Dbi;
  std::unique_ptr<TpiStream> Tpi;
};
}
}

#endif
