//===- MappedBlockStream.h - Discontiguous stream data in an Msf -*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_MSF_MAPPEDBLOCKSTREAM_H
#define LLVM_DEBUGINFO_MSF_MAPPEDBLOCKSTREAM_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/DebugInfo/Msf/IMsfStreamData.h"
#include "llvm/DebugInfo/Msf/StreamInterface.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <vector>

namespace llvm {
namespace msf {

class IMsfFile;

/// MappedBlockStream represents data stored in an Msf file into chunks of a
/// particular size (called the Block Size), and whose chunks may not be
/// necessarily contiguous.  The arrangement of these chunks within the file
/// is described by some other metadata contained within the Msf file.  In
/// the case of a standard Msf Stream, the layout of the stream's blocks
/// is described by the Msf "directory", but in the case of the directory
/// itself, the layout is described by an array at a fixed location within
/// the Msf.  MappedBlockStream provides methods for reading from and writing
/// to one of these streams transparently, as if it were a contiguous sequence
/// of bytes.
class MappedBlockStream : public StreamInterface {
public:
  Error readBytes(uint32_t Offset, uint32_t Size,
                  ArrayRef<uint8_t> &Buffer) const override;
  Error readLongestContiguousChunk(uint32_t Offset,
                                   ArrayRef<uint8_t> &Buffer) const override;
  Error writeBytes(uint32_t Offset, ArrayRef<uint8_t> Buffer) const override;

  uint32_t getLength() const override;
  Error commit() const override;

  uint32_t getNumBytesCopied() const;

  static Expected<std::unique_ptr<MappedBlockStream>>
  createIndexedStream(uint32_t StreamIdx, const IMsfFile &File);
  static Expected<std::unique_ptr<MappedBlockStream>>
  createDirectoryStream(uint32_t Length, ArrayRef<support::ulittle32_t> Blocks,
                        const IMsfFile &File);

  llvm::BumpPtrAllocator &getAllocator() { return Pool; }

protected:
  MappedBlockStream(std::unique_ptr<IMsfStreamData> Data, const IMsfFile &File);

  Error readBytes(uint32_t Offset, MutableArrayRef<uint8_t> Buffer) const;
  bool tryReadContiguously(uint32_t Offset, uint32_t Size,
                           ArrayRef<uint8_t> &Buffer) const;

  const IMsfFile &Msf;
  std::unique_ptr<IMsfStreamData> Data;

  typedef MutableArrayRef<uint8_t> CacheEntry;
  mutable llvm::BumpPtrAllocator Pool;
  mutable DenseMap<uint32_t, std::vector<CacheEntry>> CacheMap;
};

} // end namespace pdb
} // end namespace llvm

#endif // LLVM_DEBUGINFO_MSF_MAPPEDBLOCKSTREAM_H
