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
#include "llvm/DebugInfo/CodeView/StreamArray.h"
#include "llvm/DebugInfo/CodeView/StreamInterface.h"
#include "llvm/DebugInfo/CodeView/StreamReader.h"
#include "llvm/DebugInfo/CodeView/StreamWriter.h"
#include "llvm/DebugInfo/PDB/Raw/DbiStream.h"
#include "llvm/DebugInfo/PDB/Raw/DirectoryStreamData.h"
#include "llvm/DebugInfo/PDB/Raw/IndexedStreamData.h"
#include "llvm/DebugInfo/PDB/Raw/InfoStream.h"
#include "llvm/DebugInfo/PDB/Raw/NameHashTable.h"
#include "llvm/DebugInfo/PDB/Raw/PublicsStream.h"
#include "llvm/DebugInfo/PDB/Raw/RawError.h"
#include "llvm/DebugInfo/PDB/Raw/SymbolStream.h"
#include "llvm/DebugInfo/PDB/Raw/TpiStream.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::pdb;

namespace {
typedef FixedStreamArray<support::ulittle32_t> ulittle_array;
}

PDBFile::PDBFile(std::unique_ptr<StreamInterface> PdbFileBuffer)
    : Buffer(std::move(PdbFileBuffer)), SB(nullptr) {}

PDBFile::~PDBFile() {}

uint32_t PDBFile::getBlockSize() const { return SB->BlockSize; }

uint32_t PDBFile::getUnknown0() const { return SB->Unknown0; }

uint32_t PDBFile::getBlockCount() const { return SB->NumBlocks; }

uint32_t PDBFile::getNumDirectoryBytes() const { return SB->NumDirectoryBytes; }

uint32_t PDBFile::getBlockMapIndex() const { return SB->BlockMapAddr; }

uint32_t PDBFile::getUnknown1() const { return SB->Unknown1; }

uint32_t PDBFile::getNumDirectoryBlocks() const {
  return bytesToBlocks(SB->NumDirectoryBytes, SB->BlockSize);
}

uint64_t PDBFile::getBlockMapOffset() const {
  return (uint64_t)SB->BlockMapAddr * SB->BlockSize;
}

uint32_t PDBFile::getNumStreams() const { return StreamSizes.size(); }

uint32_t PDBFile::getStreamByteSize(uint32_t StreamIndex) const {
  return StreamSizes[StreamIndex];
}

ArrayRef<support::ulittle32_t>
PDBFile::getStreamBlockList(uint32_t StreamIndex) const {
  return StreamMap[StreamIndex];
}

size_t PDBFile::getFileSize() const { return Buffer->getLength(); }

ArrayRef<uint8_t> PDBFile::getBlockData(uint32_t BlockIndex,
                                        uint32_t NumBytes) const {
  uint64_t StreamBlockOffset = blockToOffset(BlockIndex, getBlockSize());

  ArrayRef<uint8_t> Result;
  if (auto EC = Buffer->readBytes(StreamBlockOffset, NumBytes, Result))
    consumeError(std::move(EC));
  return Result;
}

Error PDBFile::setBlockData(uint32_t BlockIndex, uint32_t Offset,
                            ArrayRef<uint8_t> Data) const {
  if (Offset >= getBlockSize())
    return make_error<RawError>(
        raw_error_code::invalid_block_address,
        "setBlockData attempted to write out of block bounds.");
  if (Data.size() > getBlockSize() - Offset)
    return make_error<RawError>(
        raw_error_code::invalid_block_address,
        "setBlockData attempted to write out of block bounds.");

  uint64_t StreamBlockOffset = blockToOffset(BlockIndex, getBlockSize());
  StreamBlockOffset += Offset;
  return Buffer->writeBytes(StreamBlockOffset, Data);
}

Error PDBFile::parseFileHeaders() {
  StreamReader Reader(*Buffer);

  if (auto EC = Reader.readObject(SB)) {
    consumeError(std::move(EC));
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Does not contain superblock");
  }

  if (auto EC = setSuperBlock(SB))
    return EC;

  Reader.setOffset(getBlockMapOffset());
  if (auto EC = Reader.readArray(DirectoryBlocks, getNumDirectoryBlocks()))
    return EC;

  return Error::success();
}

Error PDBFile::parseStreamData() {
  assert(SB);
  if (DirectoryStream)
    return Error::success();

  uint32_t NumStreams = 0;

  // Normally you can't use a MappedBlockStream without having fully parsed the
  // PDB file, because it accesses the directory and various other things, which
  // is exactly what we are attempting to parse.  By specifying a custom
  // subclass of IPDBStreamData which only accesses the fields that have already
  // been parsed, we can avoid this and reuse MappedBlockStream.
  auto DS = MappedBlockStream::createDirectoryStream(*this);
  if (!DS)
    return DS.takeError();
  StreamReader Reader(**DS);
  if (auto EC = Reader.readInteger(NumStreams))
    return EC;

  if (auto EC = Reader.readArray(StreamSizes, NumStreams))
    return EC;
  for (uint32_t I = 0; I < NumStreams; ++I) {
    uint32_t StreamSize = getStreamByteSize(I);
    // FIXME: What does StreamSize ~0U mean?
    uint64_t NumExpectedStreamBlocks =
        StreamSize == UINT32_MAX ? 0 : bytesToBlocks(StreamSize, SB->BlockSize);

    // For convenience, we store the block array contiguously.  This is because
    // if someone calls setStreamMap(), it is more convenient to be able to call
    // it with an ArrayRef instead of setting up a StreamRef.  Since the
    // DirectoryStream is cached in the class and thus lives for the life of the
    // class, we can be guaranteed that readArray() will return a stable
    // reference, even if it has to allocate from its internal pool.
    ArrayRef<support::ulittle32_t> Blocks;
    if (auto EC = Reader.readArray(Blocks, NumExpectedStreamBlocks))
      return EC;
    StreamMap.push_back(Blocks);
  }

  // We should have read exactly SB->NumDirectoryBytes bytes.
  assert(Reader.bytesRemaining() == 0);
  DirectoryStream = std::move(*DS);
  return Error::success();
}

llvm::ArrayRef<support::ulittle32_t> PDBFile::getDirectoryBlockArray() const {
  return DirectoryBlocks;
}

Expected<InfoStream &> PDBFile::emplacePDBInfoStream() {
  if (Info)
    Info.reset();

  auto InfoS = MappedBlockStream::createIndexedStream(StreamPDB, *this);
  if (!InfoS)
    return InfoS.takeError();
  Info = llvm::make_unique<InfoStream>(std::move(*InfoS));
  return *Info;
}

Expected<InfoStream &> PDBFile::getPDBInfoStream() {
  if (!Info) {
    auto InfoS = MappedBlockStream::createIndexedStream(StreamPDB, *this);
    if (!InfoS)
      return InfoS.takeError();
    auto TempInfo = llvm::make_unique<InfoStream>(std::move(*InfoS));
    if (auto EC = TempInfo->reload())
      return std::move(EC);
    Info = std::move(TempInfo);
  }
  return *Info;
}

Expected<DbiStream &> PDBFile::getPDBDbiStream() {
  if (!Dbi) {
    auto DbiS = MappedBlockStream::createIndexedStream(StreamDBI, *this);
    if (!DbiS)
      return DbiS.takeError();
    auto TempDbi = llvm::make_unique<DbiStream>(*this, std::move(*DbiS));
    if (auto EC = TempDbi->reload())
      return std::move(EC);
    Dbi = std::move(TempDbi);
  }
  return *Dbi;
}

Expected<TpiStream &> PDBFile::getPDBTpiStream() {
  if (!Tpi) {
    auto TpiS = MappedBlockStream::createIndexedStream(StreamTPI, *this);
    if (!TpiS)
      return TpiS.takeError();
    auto TempTpi = llvm::make_unique<TpiStream>(*this, std::move(*TpiS));
    if (auto EC = TempTpi->reload())
      return std::move(EC);
    Tpi = std::move(TempTpi);
  }
  return *Tpi;
}

Expected<TpiStream &> PDBFile::getPDBIpiStream() {
  if (!Ipi) {
    auto IpiS = MappedBlockStream::createIndexedStream(StreamIPI, *this);
    if (!IpiS)
      return IpiS.takeError();
    auto TempIpi = llvm::make_unique<TpiStream>(*this, std::move(*IpiS));
    if (auto EC = TempIpi->reload())
      return std::move(EC);
    Ipi = std::move(TempIpi);
  }
  return *Ipi;
}

Expected<PublicsStream &> PDBFile::getPDBPublicsStream() {
  if (!Publics) {
    auto DbiS = getPDBDbiStream();
    if (!DbiS)
      return DbiS.takeError();

    uint32_t PublicsStreamNum = DbiS->getPublicSymbolStreamIndex();

    auto PublicS =
        MappedBlockStream::createIndexedStream(PublicsStreamNum, *this);
    if (!PublicS)
      return PublicS.takeError();
    auto TempPublics =
        llvm::make_unique<PublicsStream>(*this, std::move(*PublicS));
    if (auto EC = TempPublics->reload())
      return std::move(EC);
    Publics = std::move(TempPublics);
  }
  return *Publics;
}

Expected<SymbolStream &> PDBFile::getPDBSymbolStream() {
  if (!Symbols) {
    auto DbiS = getPDBDbiStream();
    if (!DbiS)
      return DbiS.takeError();

    uint32_t SymbolStreamNum = DbiS->getSymRecordStreamIndex();

    auto SymbolS =
        MappedBlockStream::createIndexedStream(SymbolStreamNum, *this);
    if (!SymbolS)
      return SymbolS.takeError();
    auto TempSymbols = llvm::make_unique<SymbolStream>(std::move(*SymbolS));
    if (auto EC = TempSymbols->reload())
      return std::move(EC);
    Symbols = std::move(TempSymbols);
  }
  return *Symbols;
}

Expected<NameHashTable &> PDBFile::getStringTable() {
  if (!StringTable || !StringTableStream) {
    auto IS = getPDBInfoStream();
    if (!IS)
      return IS.takeError();

    uint32_t NameStreamIndex = IS->getNamedStreamIndex("/names");

    if (NameStreamIndex == 0)
      return make_error<RawError>(raw_error_code::no_stream);
    if (NameStreamIndex >= getNumStreams())
      return make_error<RawError>(raw_error_code::no_stream);

    auto NS = MappedBlockStream::createIndexedStream(NameStreamIndex, *this);
    if (!NS)
      return NS.takeError();

    StreamReader Reader(**NS);
    auto N = llvm::make_unique<NameHashTable>();
    if (auto EC = N->load(Reader))
      return std::move(EC);
    StringTable = std::move(N);
    StringTableStream = std::move(*NS);
  }
  return *StringTable;
}

Error PDBFile::setSuperBlock(const SuperBlock *Block) {
  SB = Block;

  // Check the magic bytes.
  if (memcmp(SB->MagicBytes, MsfMagic, sizeof(MsfMagic)) != 0)
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "MSF magic header doesn't match");

  // We don't support blocksizes which aren't a multiple of four bytes.
  if (SB->BlockSize % sizeof(support::ulittle32_t) != 0)
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Block size is not multiple of 4.");

  switch (SB->BlockSize) {
  case 512:
  case 1024:
  case 2048:
  case 4096:
    break;
  default:
    // An invalid block size suggests a corrupt PDB file.
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Unsupported block size.");
  }

  if (Buffer->getLength() % SB->BlockSize != 0)
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "File size is not a multiple of block size");

  // We don't support directories whose sizes aren't a multiple of four bytes.
  if (SB->NumDirectoryBytes % sizeof(support::ulittle32_t) != 0)
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Directory size is not multiple of 4.");

  // The number of blocks which comprise the directory is a simple function of
  // the number of bytes it contains.
  uint64_t NumDirectoryBlocks = getNumDirectoryBlocks();

  // The directory, as we understand it, is a block which consists of a list of
  // block numbers.  It is unclear what would happen if the number of blocks
  // couldn't fit on a single block.
  if (NumDirectoryBlocks > SB->BlockSize / sizeof(support::ulittle32_t))
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Too many directory blocks.");

  return Error::success();
}

void PDBFile::setStreamSizes(ArrayRef<support::ulittle32_t> Sizes) {
  StreamSizes = Sizes;
}

void PDBFile::setStreamMap(
    std::vector<ArrayRef<support::ulittle32_t>> &Streams) {
  StreamMap = Streams;
}

void PDBFile::setDirectoryBlocks(ArrayRef<support::ulittle32_t> Directory) {
  DirectoryBlocks = Directory;
}

Error PDBFile::generateSimpleStreamMap() {
  if (StreamSizes.empty())
    return Error::success();

  static std::vector<std::vector<support::ulittle32_t>> StaticMap;
  StreamMap.clear();
  StaticMap.clear();

  // Figure out how many blocks are needed for all streams, and set the first
  // used block to the highest block so that we can write the rest of the
  // blocks contiguously.
  uint32_t TotalFileBlocks = getBlockCount();
  std::vector<support::ulittle32_t> ReservedBlocks;
  ReservedBlocks.push_back(support::ulittle32_t(0));
  ReservedBlocks.push_back(SB->BlockMapAddr);
  ReservedBlocks.insert(ReservedBlocks.end(), DirectoryBlocks.begin(),
                        DirectoryBlocks.end());

  uint32_t BlocksNeeded = 0;
  for (auto Size : StreamSizes)
    BlocksNeeded += bytesToBlocks(Size, getBlockSize());

  support::ulittle32_t NextBlock(TotalFileBlocks - BlocksNeeded -
                                 ReservedBlocks.size());

  StaticMap.resize(StreamSizes.size());
  for (uint32_t S = 0; S < StreamSizes.size(); ++S) {
    uint32_t Size = StreamSizes[S];
    uint32_t NumBlocks = bytesToBlocks(Size, getBlockSize());
    auto &ThisStream = StaticMap[S];
    for (uint32_t I = 0; I < NumBlocks;) {
      NextBlock += 1;
      if (std::find(ReservedBlocks.begin(), ReservedBlocks.end(), NextBlock) !=
          ReservedBlocks.end())
        continue;

      ++I;
      assert(NextBlock < getBlockCount());
      ThisStream.push_back(NextBlock);
    }
    StreamMap.push_back(ThisStream);
  }
  return Error::success();
}

Error PDBFile::commit() {
  StreamWriter Writer(*Buffer);

  if (auto EC = Writer.writeObject(*SB))
    return EC;
  Writer.setOffset(getBlockMapOffset());
  if (auto EC = Writer.writeArray(DirectoryBlocks))
    return EC;

  auto DS = MappedBlockStream::createDirectoryStream(*this);
  if (!DS)
    return DS.takeError();
  auto DirStream = std::move(*DS);
  StreamWriter DW(*DirStream);
  if (auto EC = DW.writeInteger(this->getNumStreams()))
    return EC;

  if (auto EC = DW.writeArray(StreamSizes))
    return EC;

  for (const auto &Blocks : StreamMap) {
    if (auto EC = DW.writeArray(Blocks))
      return EC;
  }

  if (Info) {
    if (auto EC = Info->commit())
      return EC;
  }

  if (Dbi) {
    if (auto EC = Dbi->commit())
      return EC;
  }

  if (Symbols) {
    if (auto EC = Symbols->commit())
      return EC;
  }

  if (Publics) {
    if (auto EC = Publics->commit())
      return EC;
  }

  if (Tpi) {
    if (auto EC = Tpi->commit())
      return EC;
  }

  if (Ipi) {
    if (auto EC = Ipi->commit())
      return EC;
  }

  return Buffer->commit();
}