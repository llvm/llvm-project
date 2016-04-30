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

using namespace llvm;
using namespace llvm::pdb;

MappedBlockStream::MappedBlockStream(uint32_t StreamIdx, const PDBFile &File) : Pdb(File) {
  StreamLength = Pdb.getStreamByteSize(StreamIdx);
  BlockList = Pdb.getStreamBlockList(StreamIdx);
}

std::error_code
MappedBlockStream::readBytes(uint32_t Offset,
                             MutableArrayRef<uint8_t> Buffer) const {
  uint32_t BlockNum = Offset / Pdb.getBlockSize();
  uint32_t OffsetInBlock = Offset % Pdb.getBlockSize();

  // Make sure we aren't trying to read beyond the end of the stream.
  if (Buffer.size() > StreamLength)
    return std::make_error_code(std::errc::bad_address);
  if (Offset > StreamLength - Buffer.size())
    return std::make_error_code(std::errc::bad_address);

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

  return std::error_code();
}
