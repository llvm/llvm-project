//===- PDBFile.cpp - Low level interface to a PDB file ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Raw/PDBFile.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/DebugInfo/PDB/Raw/DbiStream.h"
#include "llvm/DebugInfo/PDB/Raw/InfoStream.h"
#include "llvm/DebugInfo/PDB/Raw/PublicsStream.h"
#include "llvm/DebugInfo/PDB/Raw/RawError.h"
#include "llvm/DebugInfo/PDB/Raw/SymbolStream.h"
#include "llvm/DebugInfo/PDB/Raw/TpiStream.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace llvm::pdb;

namespace {
static const char Magic[] = {'M',  'i',  'c',    'r', 'o', 's',  'o',  'f',
                             't',  ' ',  'C',    '/', 'C', '+',  '+',  ' ',
                             'M',  'S',  'F',    ' ', '7', '.',  '0',  '0',
                             '\r', '\n', '\x1a', 'D', 'S', '\0', '\0', '\0'};

// The superblock is overlaid at the beginning of the file (offset 0).
// It starts with a magic header and is followed by information which describes
// the layout of the file system.
struct SuperBlock {
  char MagicBytes[sizeof(Magic)];
  // The file system is split into a variable number of fixed size elements.
  // These elements are referred to as blocks.  The size of a block may vary
  // from system to system.
  support::ulittle32_t BlockSize;
  // This field's purpose is not yet known.
  support::ulittle32_t Unknown0;
  // This contains the number of blocks resident in the file system.  In
  // practice, NumBlocks * BlockSize is equivalent to the size of the PDB file.
  support::ulittle32_t NumBlocks;
  // This contains the number of bytes which make up the directory.
  support::ulittle32_t NumDirectoryBytes;
  // This field's purpose is not yet known.
  support::ulittle32_t Unknown1;
  // This contains the block # of the block map.
  support::ulittle32_t BlockMapAddr;
};
}

struct llvm::pdb::PDBFileContext {
  std::unique_ptr<MemoryBuffer> Buffer;
  const SuperBlock *SB;
  std::vector<uint32_t> StreamSizes;
  DenseMap<uint32_t, std::vector<uint32_t>> StreamMap;
};

static Error checkOffset(MemoryBufferRef M, uintptr_t Addr,
                         const uint64_t Size) {
  if (Addr + Size < Addr || Addr + Size < Size ||
      Addr + Size > uintptr_t(M.getBufferEnd()) ||
      Addr < uintptr_t(M.getBufferStart())) {
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Invalid buffer address");
  }
  return Error::success();
}

template <typename T>
static Error checkOffset(MemoryBufferRef M, ArrayRef<T> AR) {
  return checkOffset(M, uintptr_t(AR.data()), (uint64_t)AR.size() * sizeof(T));
}

PDBFile::PDBFile(std::unique_ptr<MemoryBuffer> MemBuffer) {
  Context.reset(new PDBFileContext());
  Context->Buffer = std::move(MemBuffer);
}

PDBFile::~PDBFile() {}

uint32_t PDBFile::getBlockSize() const { return Context->SB->BlockSize; }

uint32_t PDBFile::getUnknown0() const { return Context->SB->Unknown0; }

uint32_t PDBFile::getBlockCount() const { return Context->SB->NumBlocks; }

uint32_t PDBFile::getNumDirectoryBytes() const {
  return Context->SB->NumDirectoryBytes;
}

uint32_t PDBFile::getBlockMapIndex() const { return Context->SB->BlockMapAddr; }

uint32_t PDBFile::getUnknown1() const { return Context->SB->Unknown1; }

uint32_t PDBFile::getNumDirectoryBlocks() const {
  return bytesToBlocks(Context->SB->NumDirectoryBytes, Context->SB->BlockSize);
}

uint64_t PDBFile::getBlockMapOffset() const {
  return (uint64_t)Context->SB->BlockMapAddr * Context->SB->BlockSize;
}

uint32_t PDBFile::getNumStreams() const { return Context->StreamSizes.size(); }

uint32_t PDBFile::getStreamByteSize(uint32_t StreamIndex) const {
  return Context->StreamSizes[StreamIndex];
}

llvm::ArrayRef<uint32_t>
PDBFile::getStreamBlockList(uint32_t StreamIndex) const {
  auto &Data = Context->StreamMap[StreamIndex];
  return llvm::ArrayRef<uint32_t>(Data);
}

StringRef PDBFile::getBlockData(uint32_t BlockIndex, uint32_t NumBytes) const {
  uint64_t StreamBlockOffset = blockToOffset(BlockIndex, getBlockSize());

  return StringRef(Context->Buffer->getBufferStart() + StreamBlockOffset,
                   NumBytes);
}

Error PDBFile::parseFileHeaders() {
  std::error_code EC;
  MemoryBufferRef BufferRef = *Context->Buffer;

  // Make sure the file is sufficiently large to hold a super block.
  // Do this before attempting to read the super block.
  if (BufferRef.getBufferSize() < sizeof(SuperBlock))
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Does not contain superblock");

  Context->SB =
      reinterpret_cast<const SuperBlock *>(BufferRef.getBufferStart());
  const SuperBlock *SB = Context->SB;
  // Check the magic bytes.
  if (memcmp(SB->MagicBytes, Magic, sizeof(Magic)) != 0)
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "MSF magic header doesn't match");

  // We don't support blocksizes which aren't a multiple of four bytes.
  if (SB->BlockSize % sizeof(support::ulittle32_t) != 0)
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Block size is not multiple of 4.");

  switch (SB->BlockSize) {
  case 512: case 1024: case 2048: case 4096:
    break;
  default:
    // An invalid block size suggests a corrupt PDB file.
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Unsupported block size.");
  }

  if (BufferRef.getBufferSize() % SB->BlockSize != 0)
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "File size is not a multiple of block size");

  // We don't support directories whose sizes aren't a multiple of four bytes.
  if (SB->NumDirectoryBytes % sizeof(support::ulittle32_t) != 0)
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Directory size is not multiple of 4.");

  // The number of blocks which comprise the directory is a simple function of
  // the number of bytes it contains.
  uint64_t NumDirectoryBlocks = getNumDirectoryBlocks();

  // The block map, as we understand it, is a block which consists of a list of
  // block numbers.
  // It is unclear what would happen if the number of blocks couldn't fit on a
  // single block.
  if (NumDirectoryBlocks > SB->BlockSize / sizeof(support::ulittle32_t))
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Too many directory blocks.");

  // Make sure the directory block array fits within the file.
  if (auto EC = checkOffset(BufferRef, getDirectoryBlockArray()))
    return EC;

  return Error::success();
}

Error PDBFile::parseStreamData() {
  assert(Context && Context->SB);

  bool SeenNumStreams = false;
  uint32_t NumStreams = 0;
  uint32_t StreamIdx = 0;
  uint64_t DirectoryBytesRead = 0;

  MemoryBufferRef M = *Context->Buffer;
  const SuperBlock *SB = Context->SB;

  auto DirectoryBlocks = getDirectoryBlockArray();

  // The structure of the directory is as follows:
  //    struct PDBDirectory {
  //      uint32_t NumStreams;
  //      uint32_t StreamSizes[NumStreams];
  //      uint32_t StreamMap[NumStreams][];
  //    };
  //
  //  Empty streams don't consume entries in the StreamMap.
  for (uint32_t DirectoryBlockAddr : DirectoryBlocks) {
    uint64_t DirectoryBlockOffset =
        blockToOffset(DirectoryBlockAddr, SB->BlockSize);
    auto DirectoryBlock =
        makeArrayRef(reinterpret_cast<const support::ulittle32_t *>(
                         M.getBufferStart() + DirectoryBlockOffset),
                     SB->BlockSize / sizeof(support::ulittle32_t));
    if (auto EC = checkOffset(M, DirectoryBlock))
      return EC;

    // We read data out of the directory four bytes at a time.  Depending on
    // where we are in the directory, the contents may be: the number of streams
    // in the directory, a stream's size, or a block in the stream map.
    for (uint32_t Data : DirectoryBlock) {
      // Don't read beyond the end of the directory.
      if (DirectoryBytesRead == SB->NumDirectoryBytes)
        break;

      DirectoryBytesRead += sizeof(Data);

      // This data must be the number of streams if we haven't seen it yet.
      if (!SeenNumStreams) {
        NumStreams = Data;
        SeenNumStreams = true;
        continue;
      }
      // This data must be a stream size if we have not seen them all yet.
      if (Context->StreamSizes.size() < NumStreams) {
        // It seems like some streams have their set to -1 when their contents
        // are not present.  Treat them like empty streams for now.
        if (Data == UINT32_MAX)
          Context->StreamSizes.push_back(0);
        else
          Context->StreamSizes.push_back(Data);
        continue;
      }

      // This data must be a stream block number if we have seen all of the
      // stream sizes.
      std::vector<uint32_t> *StreamBlocks = nullptr;
      // Figure out which stream this block number belongs to.
      while (StreamIdx < NumStreams) {
        uint64_t NumExpectedStreamBlocks =
            bytesToBlocks(Context->StreamSizes[StreamIdx], SB->BlockSize);
        StreamBlocks = &Context->StreamMap[StreamIdx];
        if (NumExpectedStreamBlocks > StreamBlocks->size())
          break;
        ++StreamIdx;
      }
      // It seems this block doesn't belong to any stream?  The stream is either
      // corrupt or something more mysterious is going on.
      if (StreamIdx == NumStreams)
        return make_error<RawError>(raw_error_code::corrupt_file,
                                    "Orphaned block found?");

      uint64_t BlockOffset = blockToOffset(Data, getBlockSize());
      if (BlockOffset + getBlockSize() < BlockOffset)
        return make_error<RawError>(raw_error_code::corrupt_file,
                                    "Bogus stream block number");
      if (BlockOffset + getBlockSize() > M.getBufferSize())
        return make_error<RawError>(raw_error_code::corrupt_file,
                                    "Stream block number is out of bounds");

      StreamBlocks->push_back(Data);
    }
  }

  if (Context->StreamSizes.size() != NumStreams)
    return make_error<RawError>(
        raw_error_code::corrupt_file,
        "The directory has fewer streams then expected");

  for (uint32_t I = 0; I != NumStreams; ++I) {
    uint64_t NumExpectedStreamBlocks =
        bytesToBlocks(getStreamByteSize(I), getBlockSize());
    size_t NumStreamBlocks = getStreamBlockList(I).size();
    if (NumExpectedStreamBlocks != NumStreamBlocks)
      return make_error<RawError>(raw_error_code::corrupt_file,
                                  "The number of stream blocks is not "
                                  "sufficient for the size of this stream");
  }

  // We should have read exactly SB->NumDirectoryBytes bytes.
  assert(DirectoryBytesRead == SB->NumDirectoryBytes);
  return Error::success();
}

llvm::ArrayRef<support::ulittle32_t> PDBFile::getDirectoryBlockArray() {
  return makeArrayRef(
      reinterpret_cast<const support::ulittle32_t *>(
          Context->Buffer->getBufferStart() + getBlockMapOffset()),
      getNumDirectoryBlocks());
}

Expected<InfoStream &> PDBFile::getPDBInfoStream() {
  if (!Info) {
    Info.reset(new InfoStream(*this));
    if (auto EC = Info->reload())
      return std::move(EC);
  }
  return *Info;
}

Expected<DbiStream &> PDBFile::getPDBDbiStream() {
  if (!Dbi) {
    Dbi.reset(new DbiStream(*this));
    if (auto EC = Dbi->reload())
      return std::move(EC);
  }
  return *Dbi;
}

Expected<TpiStream &> PDBFile::getPDBTpiStream() {
  if (!Tpi) {
    Tpi.reset(new TpiStream(*this, StreamTPI));
    if (auto EC = Tpi->reload())
      return std::move(EC);
  }
  return *Tpi;
}

Expected<TpiStream &> PDBFile::getPDBIpiStream() {
  if (!Ipi) {
    Ipi.reset(new TpiStream(*this, StreamIPI));
    if (auto EC = Ipi->reload())
      return std::move(EC);
  }
  return *Ipi;
}

Expected<PublicsStream &> PDBFile::getPDBPublicsStream() {
  if (!Publics) {
    auto DbiS = getPDBDbiStream();
    if (auto EC = DbiS.takeError())
      return std::move(EC);
    uint32_t PublicsStreamNum = DbiS->getPublicSymbolStreamIndex();

    Publics.reset(new PublicsStream(*this, PublicsStreamNum));
    if (auto EC = Publics->reload())
      return std::move(EC);
  }
  return *Publics;
}

Expected<SymbolStream &> PDBFile::getPDBSymbolStream() {
  if (!Symbols) {
    auto DbiS = getPDBDbiStream();
    if (auto EC = DbiS.takeError())
      return std::move(EC);
    uint32_t SymbolStreamNum = DbiS->getSymRecordStreamIndex();

    Symbols.reset(new SymbolStream(*this, SymbolStreamNum));
    if (auto EC = Symbols->reload())
      return std::move(EC);
  }
  return *Symbols;
}
