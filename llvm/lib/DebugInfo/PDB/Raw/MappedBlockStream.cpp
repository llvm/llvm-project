//===- MappedBlockStream.cpp - Reads stream data from a PDBFile -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Raw/MappedBlockStream.h"
#include "llvm/DebugInfo/PDB/Raw/PDBFile.h"
#include "llvm/DebugInfo/PDB/Raw/RawError.h"

using namespace llvm;
using namespace llvm::pdb;

MappedBlockStream::MappedBlockStream(uint32_t StreamIdx, const PDBFile &File) : Pdb(File) {
  if (StreamIdx >= Pdb.getNumStreams()) {
    StreamLength = 0;
  } else {
    StreamLength = Pdb.getStreamByteSize(StreamIdx);
    BlockList = Pdb.getStreamBlockList(StreamIdx);
  }
}

Error MappedBlockStream::readBytes(uint32_t Offset,
                                   MutableArrayRef<uint8_t> Buffer) const {
  uint32_t BlockNum = Offset / Pdb.getBlockSize();
  uint32_t OffsetInBlock = Offset % Pdb.getBlockSize();

  // Make sure we aren't trying to read beyond the end of the stream.
  if (Buffer.size() > StreamLength)
    return make_error<RawError>(raw_error_code::insufficient_buffer);
  if (Offset > StreamLength - Buffer.size())
    return make_error<RawError>(raw_error_code::insufficient_buffer);

  uint32_t BytesLeft = Buffer.size();
  uint32_t BytesWritten = 0;
  uint8_t *WriteBuffer = Buffer.data();
  while (BytesLeft > 0) {
    uint32_t StreamBlockAddr = BlockList[BlockNum];

    StringRef Data = Pdb.getBlockData(StreamBlockAddr, Pdb.getBlockSize());

    const char *ChunkStart = Data.data() + OffsetInBlock;
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

Error MappedBlockStream::getArrayRef(uint32_t Offset, ArrayRef<uint8_t> &Buffer,
                                     uint32_t Length) const {
  uint32_t BlockNum = Offset / Pdb.getBlockSize();
  uint32_t OffsetInBlock = Offset % Pdb.getBlockSize();
  uint32_t BytesAvailableInBlock = Pdb.getBlockSize() - OffsetInBlock;

  // If this is the last block in the stream, not all of the data is valid.
  if (BlockNum == BlockList.size() - 1) {
    uint32_t AllocatedBytesInBlock = StreamLength % Pdb.getBlockSize();
    if (AllocatedBytesInBlock < BytesAvailableInBlock)
      BytesAvailableInBlock = AllocatedBytesInBlock;
  }
  if (BytesAvailableInBlock < Length)
    return make_error<RawError>(raw_error_code::feature_unsupported);

  uint32_t StreamBlockAddr = BlockList[BlockNum];
  StringRef Data = Pdb.getBlockData(StreamBlockAddr, Pdb.getBlockSize());
  Data = Data.substr(OffsetInBlock, Length);

  Buffer = ArrayRef<uint8_t>(Data.bytes_begin(), Data.bytes_end());
  return Error::success();
}
