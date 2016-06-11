//===- MappedBlockStream.cpp - Reads stream data from a PDBFile -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Raw/MappedBlockStream.h"
#include "llvm/DebugInfo/PDB/Raw/DirectoryStreamData.h"
#include "llvm/DebugInfo/PDB/Raw/IPDBStreamData.h"
#include "llvm/DebugInfo/PDB/Raw/IndexedStreamData.h"
#include "llvm/DebugInfo/PDB/Raw/PDBFile.h"
#include "llvm/DebugInfo/PDB/Raw/RawError.h"

using namespace llvm;
using namespace llvm::pdb;

namespace {
// This exists so that we can use make_unique while still keeping the
// constructor of MappedBlockStream private, forcing users to go through
// the `create` interface.
class MappedBlockStreamImpl : public MappedBlockStream {
public:
  MappedBlockStreamImpl(std::unique_ptr<IPDBStreamData> Data,
                        const IPDBFile &File)
      : MappedBlockStream(std::move(Data), File) {}
};
}

typedef std::pair<uint32_t, uint32_t> Interval;
static Interval intersect(const Interval &I1, const Interval &I2) {
  return std::make_pair(std::max(I1.first, I2.first),
                        std::min(I1.second, I2.second));
}

MappedBlockStream::MappedBlockStream(std::unique_ptr<IPDBStreamData> Data,
                                     const IPDBFile &Pdb)
    : Pdb(Pdb), Data(std::move(Data)) {}

Error MappedBlockStream::readBytes(uint32_t Offset, uint32_t Size,
                                   ArrayRef<uint8_t> &Buffer) const {
  // Make sure we aren't trying to read beyond the end of the stream.
  if (Size > Data->getLength())
    return make_error<RawError>(raw_error_code::insufficient_buffer);
  if (Offset > Data->getLength() - Size)
    return make_error<RawError>(raw_error_code::insufficient_buffer);

  if (tryReadContiguously(Offset, Size, Buffer))
    return Error::success();

  auto CacheIter = CacheMap.find(Offset);
  if (CacheIter != CacheMap.end()) {
    // Try to find an alloc that was large enough for this request.
    for (auto &Entry : CacheIter->second) {
      if (Entry.size() >= Size) {
        Buffer = Entry.slice(0, Size);
        return Error::success();
      }
    }
  }

  // We couldn't find a buffer that started at the correct offset (the most
  // common scenario).  Try to see if there is a buffer that starts at some
  // other offset but overlaps the desired range.
  for (auto &CacheItem : CacheMap) {
    Interval RequestExtent = std::make_pair(Offset, Offset + Size);

    // We already checked this one on the fast path above.
    if (CacheItem.first == Offset)
      continue;
    // If the initial extent of the cached item is beyond the ending extent
    // of the request, there is no overlap.
    if (CacheItem.first >= Offset + Size)
      continue;

    // We really only have to check the last item in the list, since we append
    // in order of increasing length.
    if (CacheItem.second.empty())
      continue;

    auto CachedAlloc = CacheItem.second.back();
    // If the initial extent of the request is beyond the ending extent of
    // the cached item, there is no overlap.
    Interval CachedExtent =
        std::make_pair(CacheItem.first, CacheItem.first + CachedAlloc.size());
    if (RequestExtent.first >= CachedExtent.first + CachedExtent.second)
      continue;

    Interval Intersection = intersect(CachedExtent, RequestExtent);
    // Only use this if the entire request extent is contained in the cached
    // extent.
    if (Intersection != RequestExtent)
      continue;

    uint32_t CacheRangeOffset =
        AbsoluteDifference(CachedExtent.first, Intersection.first);
    Buffer = CachedAlloc.slice(CacheRangeOffset, Size);
    return Error::success();
  }

  // Otherwise allocate a large enough buffer in the pool, memcpy the data
  // into it, and return an ArrayRef to that.  Do not touch existing pool
  // allocations, as existing clients may be holding a pointer which must
  // not be invalidated.
  uint8_t *WriteBuffer = static_cast<uint8_t *>(Pool.Allocate(Size, 8));
  if (auto EC = readBytes(Offset, MutableArrayRef<uint8_t>(WriteBuffer, Size)))
    return EC;

  if (CacheIter != CacheMap.end()) {
    CacheIter->second.emplace_back(WriteBuffer, Size);
  } else {
    std::vector<CacheEntry> List;
    List.emplace_back(WriteBuffer, Size);
    CacheMap.insert(std::make_pair(Offset, List));
  }
  Buffer = ArrayRef<uint8_t>(WriteBuffer, Size);
  return Error::success();
}

Error MappedBlockStream::readLongestContiguousChunk(
    uint32_t Offset, ArrayRef<uint8_t> &Buffer) const {
  // Make sure we aren't trying to read beyond the end of the stream.
  if (Offset >= Data->getLength())
    return make_error<RawError>(raw_error_code::insufficient_buffer);
  uint32_t First = Offset / Pdb.getBlockSize();
  uint32_t Last = First;

  auto BlockList = Data->getStreamBlocks();
  while (Last < Pdb.getBlockCount() - 1) {
    if (BlockList[Last] != BlockList[Last + 1] - 1)
      break;
    ++Last;
  }

  uint32_t OffsetInFirstBlock = Offset % Pdb.getBlockSize();
  uint32_t BytesFromFirstBlock = Pdb.getBlockSize() - OffsetInFirstBlock;
  uint32_t BlockSpan = Last - First + 1;
  uint32_t ByteSpan =
      BytesFromFirstBlock + (BlockSpan - 1) * Pdb.getBlockSize();
  Buffer = Pdb.getBlockData(BlockList[First], Pdb.getBlockSize());
  Buffer = Buffer.drop_front(OffsetInFirstBlock);
  Buffer = ArrayRef<uint8_t>(Buffer.data(), ByteSpan);
  return Error::success();
}

uint32_t MappedBlockStream::getLength() const { return Data->getLength(); }

bool MappedBlockStream::tryReadContiguously(uint32_t Offset, uint32_t Size,
                                            ArrayRef<uint8_t> &Buffer) const {
  // Attempt to fulfill the request with a reference directly into the stream.
  // This can work even if the request crosses a block boundary, provided that
  // all subsequent blocks are contiguous.  For example, a 10k read with a 4k
  // block size can be filled with a reference if, from the starting offset,
  // 3 blocks in a row are contiguous.
  uint32_t BlockNum = Offset / Pdb.getBlockSize();
  uint32_t OffsetInBlock = Offset % Pdb.getBlockSize();
  uint32_t BytesFromFirstBlock =
      std::min(Size, Pdb.getBlockSize() - OffsetInBlock);
  uint32_t NumAdditionalBlocks =
      llvm::alignTo(Size - BytesFromFirstBlock, Pdb.getBlockSize()) /
      Pdb.getBlockSize();

  auto BlockList = Data->getStreamBlocks();
  uint32_t RequiredContiguousBlocks = NumAdditionalBlocks + 1;
  uint32_t E = BlockList[BlockNum];
  for (uint32_t I = 0; I < RequiredContiguousBlocks; ++I, ++E) {
    if (BlockList[I + BlockNum] != E)
      return false;
  }

  uint32_t FirstBlockAddr = BlockList[BlockNum];
  auto Data = Pdb.getBlockData(FirstBlockAddr, Pdb.getBlockSize());
  Data = Data.drop_front(OffsetInBlock);
  Buffer = ArrayRef<uint8_t>(Data.data(), Size);
  return true;
}

Error MappedBlockStream::readBytes(uint32_t Offset,
                                   MutableArrayRef<uint8_t> Buffer) const {
  uint32_t BlockNum = Offset / Pdb.getBlockSize();
  uint32_t OffsetInBlock = Offset % Pdb.getBlockSize();

  // Make sure we aren't trying to read beyond the end of the stream.
  if (Buffer.size() > Data->getLength())
    return make_error<RawError>(raw_error_code::insufficient_buffer);
  if (Offset > Data->getLength() - Buffer.size())
    return make_error<RawError>(raw_error_code::insufficient_buffer);

  uint32_t BytesLeft = Buffer.size();
  uint32_t BytesWritten = 0;
  uint8_t *WriteBuffer = Buffer.data();
  auto BlockList = Data->getStreamBlocks();
  while (BytesLeft > 0) {
    uint32_t StreamBlockAddr = BlockList[BlockNum];

    auto Data = Pdb.getBlockData(StreamBlockAddr, Pdb.getBlockSize());

    const uint8_t *ChunkStart = Data.data() + OffsetInBlock;
    uint32_t BytesInChunk =
        std::min(BytesLeft, Pdb.getBlockSize() - OffsetInBlock);
    ::memcpy(WriteBuffer + BytesWritten, ChunkStart, BytesInChunk);

    BytesWritten += BytesInChunk;
    BytesLeft -= BytesInChunk;
    ++BlockNum;
    OffsetInBlock = 0;
  }

  return Error::success();
}

Error MappedBlockStream::writeBytes(uint32_t Offset,
                                    ArrayRef<uint8_t> Buffer) const {
  // Make sure we aren't trying to write beyond the end of the stream.
  if (Buffer.size() > Data->getLength())
    return make_error<RawError>(raw_error_code::insufficient_buffer);

  if (Offset > Data->getLength() - Buffer.size())
    return make_error<RawError>(raw_error_code::insufficient_buffer);

  uint32_t BlockNum = Offset / Pdb.getBlockSize();
  uint32_t OffsetInBlock = Offset % Pdb.getBlockSize();

  uint32_t BytesLeft = Buffer.size();
  auto BlockList = Data->getStreamBlocks();
  uint32_t BytesWritten = 0;
  while (BytesLeft > 0) {
    uint32_t StreamBlockAddr = BlockList[BlockNum];
    uint32_t BytesToWriteInChunk =
        std::min(BytesLeft, Pdb.getBlockSize() - OffsetInBlock);

    const uint8_t *Chunk = Buffer.data() + BytesWritten;
    ArrayRef<uint8_t> ChunkData(Chunk, BytesToWriteInChunk);
    if (auto EC = Pdb.setBlockData(StreamBlockAddr, OffsetInBlock, ChunkData))
      return EC;

    BytesLeft -= BytesToWriteInChunk;
    BytesWritten += BytesToWriteInChunk;
    ++BlockNum;
    OffsetInBlock = 0;
  }

  // If this write overlapped a read which previously came from the pool,
  // someone may still be holding a pointer to that alloc which is now invalid.
  // Compute the overlapping range and update the cache entry, so any
  // outstanding buffers are automatically updated.
  for (const auto &MapEntry : CacheMap) {
    // If the end of the written extent precedes the beginning of the cached
    // extent, ignore this map entry.
    if (Offset + BytesWritten < MapEntry.first)
      continue;
    for (const auto &Alloc : MapEntry.second) {
      // If the end of the cached extent precedes the beginning of the written
      // extent, ignore this alloc.
      if (MapEntry.first + Alloc.size() < Offset)
        continue;

      // If we get here, they are guaranteed to overlap.
      Interval WriteInterval = std::make_pair(Offset, Offset + BytesWritten);
      Interval CachedInterval =
          std::make_pair(MapEntry.first, MapEntry.first + Alloc.size());
      // If they overlap, we need to write the new data into the overlapping
      // range.
      auto Intersection = intersect(WriteInterval, CachedInterval);
      assert(Intersection.first <= Intersection.second);

      uint32_t Length = Intersection.second - Intersection.first;
      uint32_t SrcOffset =
          AbsoluteDifference(WriteInterval.first, Intersection.first);
      uint32_t DestOffset =
          AbsoluteDifference(CachedInterval.first, Intersection.first);
      ::memcpy(Alloc.data() + DestOffset, Buffer.data() + SrcOffset, Length);
    }
  }

  return Error::success();
}

uint32_t MappedBlockStream::getNumBytesCopied() const {
  return static_cast<uint32_t>(Pool.getBytesAllocated());
}

Expected<std::unique_ptr<MappedBlockStream>>
MappedBlockStream::createIndexedStream(uint32_t StreamIdx,
                                       const IPDBFile &File) {
  if (StreamIdx >= File.getNumStreams())
    return make_error<RawError>(raw_error_code::no_stream);

  auto Data = llvm::make_unique<IndexedStreamData>(StreamIdx, File);
  return llvm::make_unique<MappedBlockStreamImpl>(std::move(Data), File);
}

Expected<std::unique_ptr<MappedBlockStream>>
MappedBlockStream::createDirectoryStream(const PDBFile &File) {
  auto Data = llvm::make_unique<DirectoryStreamData>(File);
  return llvm::make_unique<MappedBlockStreamImpl>(std::move(Data), File);
}
