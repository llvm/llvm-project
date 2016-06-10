//===- MappedBlockStream.h - Reads stream data from a PDBFile ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_MAPPEDBLOCKSTREAM_H
#define LLVM_DEBUGINFO_PDB_RAW_MAPPEDBLOCKSTREAM_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/DebugInfo/CodeView/StreamInterface.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <vector>

namespace llvm {
namespace pdb {

class IPDBFile;
class IPDBStreamData;
class PDBFile;

class MappedBlockStream : public codeview::StreamInterface {
public:
  Error readBytes(uint32_t Offset, uint32_t Size,
                  ArrayRef<uint8_t> &Buffer) const override;
  Error readLongestContiguousChunk(uint32_t Offset,
                                   ArrayRef<uint8_t> &Buffer) const override;
  Error writeBytes(uint32_t Offset, ArrayRef<uint8_t> Buffer) const override;

  uint32_t getLength() const override;

  uint32_t getNumBytesCopied() const;

  static Expected<std::unique_ptr<MappedBlockStream>>
  createIndexedStream(uint32_t StreamIdx, const IPDBFile &File);
  static Expected<std::unique_ptr<MappedBlockStream>>
  createDirectoryStream(const PDBFile &File);

protected:
  MappedBlockStream(std::unique_ptr<IPDBStreamData> Data, const IPDBFile &File);

  Error readBytes(uint32_t Offset, MutableArrayRef<uint8_t> Buffer) const;
  bool tryReadContiguously(uint32_t Offset, uint32_t Size,
                           ArrayRef<uint8_t> &Buffer) const;

  const IPDBFile &Pdb;
  std::unique_ptr<IPDBStreamData> Data;

  typedef MutableArrayRef<uint8_t> CacheEntry;
  mutable llvm::BumpPtrAllocator Pool;
  mutable DenseMap<uint32_t, std::vector<CacheEntry>> CacheMap;
};

} // end namespace pdb
} // end namespace llvm

#endif // LLVM_DEBUGINFO_PDB_RAW_MAPPEDBLOCKSTREAM_H
