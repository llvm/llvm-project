//===- PDBStream.cpp - Low level interface to a PDB stream ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Raw/PDBStream.h"
#include "llvm/DebugInfo/PDB/Raw/PDBFile.h"

using namespace llvm;

PDBStream::PDBStream(uint32_t StreamIdx, const PDBFile &File) : Pdb(File) {
  this->StreamLength = Pdb.getStreamByteSize(StreamIdx);
  this->BlockList = Pdb.getStreamBlockList(StreamIdx);
  this->Offset = 0;
}

std::error_code PDBStream::readInteger(uint32_t &Dest) {
  support::ulittle32_t P;
  if (std::error_code EC = readObject(&P))
    return EC;
  Dest = P;
  return std::error_code();
}

std::error_code PDBStream::readZeroString(std::string &Dest) {
  char C;
  do {
    readObject(&C);
    if (C != '\0')
      Dest.push_back(C);
  } while (C != '\0');
  return std::error_code();
}

std::error_code PDBStream::readBytes(void *Dest, uint32_t Length) {
  uint32_t BlockNum = Offset / Pdb.getBlockSize();
  uint32_t OffsetInBlock = Offset % Pdb.getBlockSize();

  // Make sure we aren't trying to read beyond the end of the stream.
  if (this->Offset + Length > this->StreamLength)
    return std::make_error_code(std::errc::bad_address);

  // Modify the passed in offset to point to the data after the object.
  Offset += Length;

  // Handle the contiguous case: the offset + size stays within a block.
  if (OffsetInBlock + Length <= Pdb.getBlockSize()) {
    uint32_t StreamBlockAddr = this->BlockList[BlockNum];

    StringRef Data = Pdb.getBlockData(StreamBlockAddr, Pdb.getBlockSize());
    ::memcpy(Dest, Data.data() + OffsetInBlock, Length);
    return std::error_code();
  }

  // The non-contiguous case: we will stitch together non-contiguous chunks
  uint32_t BytesLeft = Length;
  uint32_t BytesWritten = 0;
  char *WriteBuffer = static_cast<char *>(Dest);
  while (BytesLeft > 0) {
    uint32_t StreamBlockAddr = this->BlockList[BlockNum];
    uint64_t StreamBlockOffset =
        PDBFile::blockToOffset(StreamBlockAddr, Pdb.getBlockSize()) +
        OffsetInBlock;

    StringRef Data = Pdb.getBlockData(StreamBlockAddr, Pdb.getBlockSize());

    const char *ChunkStart = Data.data() + StreamBlockOffset;
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

void PDBStream::setOffset(uint32_t O) { this->Offset = O; }

uint32_t PDBStream::getOffset() const { return this->Offset; }

uint32_t PDBStream::getLength() const { return this->StreamLength; }
