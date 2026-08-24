//===- PDBFile.cpp - Low level interface to a PDB file ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/DebugInfo/MSF/MSFCommon.h"
#include "llvm/DebugInfo/MSF/MSFZ.h"
#include "llvm/DebugInfo/MSF/MappedBlockStream.h"
#include "llvm/DebugInfo/PDB/Native/DbiStream.h"
#include "llvm/DebugInfo/PDB/Native/GlobalsStream.h"
#include "llvm/DebugInfo/PDB/Native/InfoStream.h"
#include "llvm/DebugInfo/PDB/Native/InjectedSourceStream.h"
#include "llvm/DebugInfo/PDB/Native/PDBStringTable.h"
#include "llvm/DebugInfo/PDB/Native/PublicsStream.h"
#include "llvm/DebugInfo/PDB/Native/RawError.h"
#include "llvm/DebugInfo/PDB/Native/SymbolStream.h"
#include "llvm/DebugInfo/PDB/Native/TpiStream.h"
#include "llvm/Support/BinaryStream.h"
#include "llvm/Support/BinaryStreamArray.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::msf;
using namespace llvm::pdb;

namespace {
typedef FixedStreamArray<support::ulittle32_t> ulittle_array;
} // end anonymous namespace

PDBFile::PDBFile(StringRef Path, std::unique_ptr<BinaryStream> PdbFileBuffer,
                 BumpPtrAllocator &Allocator)
    : FilePath(std::string(Path)), Allocator(Allocator),
      Buffer(std::move(PdbFileBuffer)) {}

PDBFile::~PDBFile() = default;

StringRef PDBFile::getFilePath() const { return FilePath; }

StringRef PDBFile::getFileDirectory() const {
  return sys::path::parent_path(FilePath);
}

uint32_t PDBFile::getBlockSize() const {
  return ContainerLayout.SB ? ContainerLayout.SB->BlockSize : 0;
}

uint32_t PDBFile::getFreeBlockMapBlock() const {
  return ContainerLayout.SB ? ContainerLayout.SB->FreeBlockMapBlock : 0;
}

uint32_t PDBFile::getBlockCount() const {
  return ContainerLayout.SB ? ContainerLayout.SB->NumBlocks : 0;
}

uint32_t PDBFile::getNumDirectoryBytes() const {
  return ContainerLayout.SB ? ContainerLayout.SB->NumDirectoryBytes : 0;
}

uint32_t PDBFile::getBlockMapIndex() const {
  return ContainerLayout.SB ? ContainerLayout.SB->BlockMapAddr : 0;
}

uint32_t PDBFile::getUnknown1() const {
  return ContainerLayout.SB ? ContainerLayout.SB->Unknown1 : 0;
}

uint32_t PDBFile::getNumDirectoryBlocks() const {
  if (!ContainerLayout.SB)
    return 0;
  return msf::bytesToBlocks(ContainerLayout.SB->NumDirectoryBytes,
                            ContainerLayout.SB->BlockSize);
}

uint64_t PDBFile::getBlockMapOffset() const {
  if (!ContainerLayout.SB)
    return 0;
  return (uint64_t)ContainerLayout.SB->BlockMapAddr *
         ContainerLayout.SB->BlockSize;
}

uint32_t PDBFile::getNumStreams() const {
#if LLVM_ENABLE_ZSTD
  if (MSFZ)
    return MSFZ->getNumStreams();
#endif
  return ContainerLayout.StreamSizes.size();
}

uint32_t PDBFile::getMaxStreamSize() const {
#if LLVM_ENABLE_ZSTD
  if (MSFZ) {
    uint32_t MaxSize = 0;
    for (uint32_t I = 0; I != getNumStreams(); ++I)
      MaxSize = std::max(MaxSize, getStreamByteSize(I));
    return MaxSize;
  }
#endif
  return *llvm::max_element(ContainerLayout.StreamSizes);
}

uint32_t PDBFile::getStreamByteSize(uint32_t StreamIndex) const {
#if LLVM_ENABLE_ZSTD
  if (MSFZ)
    return MSFZ->getStreamByteSize(StreamIndex);
#endif
  return ContainerLayout.StreamSizes[StreamIndex];
}

ArrayRef<support::ulittle32_t>
PDBFile::getStreamBlockList(uint32_t StreamIndex) const {
#if LLVM_ENABLE_ZSTD
  if (MSFZ)
    return {};
#endif
  return ContainerLayout.StreamMap[StreamIndex];
}

uint64_t PDBFile::getFileSize() const { return Buffer->getLength(); }

Expected<ArrayRef<uint8_t>> PDBFile::getBlockData(uint32_t BlockIndex,
                                                  uint32_t NumBytes) const {
#if LLVM_ENABLE_ZSTD
  if (MSFZ)
    return make_error<RawError>(raw_error_code::feature_unsupported,
                                "MSFZ files do not expose physical MSF blocks");
#endif
  uint64_t StreamBlockOffset = msf::blockToOffset(BlockIndex, getBlockSize());

  ArrayRef<uint8_t> Result;
  if (auto EC = Buffer->readBytes(StreamBlockOffset, NumBytes, Result))
    return std::move(EC);
  return Result;
}

Error PDBFile::setBlockData(uint32_t BlockIndex, uint32_t Offset,
                            ArrayRef<uint8_t> Data) const {
  return make_error<RawError>(raw_error_code::not_writable,
                              "PDBFile is immutable");
}

Error PDBFile::parseFileHeaders() {
  if (Buffer->getLength() >= sizeof(msf::MSFZMagic)) {
    ArrayRef<uint8_t> Magic;
    if (Error E = Buffer->readBytes(0, sizeof(msf::MSFZMagic), Magic))
      return E;
    if (msf::isMSFZMagic(Magic)) {
#if LLVM_ENABLE_ZSTD
      auto MSFZOrErr = msf::MSFZFile::create(*Buffer, Allocator);
      if (!MSFZOrErr)
        return MSFZOrErr.takeError();
      MSFZ = std::move(*MSFZOrErr);
      return Error::success();
#else
      return make_error<RawError>(raw_error_code::feature_unsupported,
                                  "MSFZ input requires LLVM_ENABLE_ZSTD");
#endif
    }
  }

  BinaryStreamReader Reader(*Buffer);

  // Initialize SB.
  const msf::SuperBlock *SB = nullptr;
  if (auto EC = Reader.readObject(SB)) {
    consumeError(std::move(EC));
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "MSF superblock is missing");
  }

  if (auto EC = msf::validateSuperBlock(*SB))
    return EC;

  if (Buffer->getLength() % SB->BlockSize != 0)
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "File size is not a multiple of block size");
  ContainerLayout.SB = SB;

  // Initialize Free Page Map.
  ContainerLayout.FreePageMap.resize(SB->NumBlocks);
  // The Fpm exists either at block 1 or block 2 of the MSF.  However, this
  // allows for a maximum of getBlockSize() * 8 blocks bits in the Fpm, and
  // thusly an equal number of total blocks in the file.  For a block size
  // of 4KiB (very common), this would yield 32KiB total blocks in file, for a
  // maximum file size of 32KiB * 4KiB = 128MiB.  Obviously this won't do, so
  // the Fpm is split across the file at `getBlockSize()` intervals.  As a
  // result, every block whose index is of the form |{1,2} + getBlockSize() * k|
  // for any non-negative integer k is an Fpm block.  In theory, we only really
  // need to reserve blocks of the form |{1,2} + getBlockSize() * 8 * k|, but
  // current versions of the MSF format already expect the Fpm to be arranged
  // at getBlockSize() intervals, so we have to be compatible.
  // See the function fpmPn() for more information:
  // https://github.com/Microsoft/microsoft-pdb/blob/master/PDB/msf/msf.cpp#L489
  auto FpmStream =
      MappedBlockStream::createFpmStream(ContainerLayout, *Buffer, Allocator);
  BinaryStreamReader FpmReader(*FpmStream);
  ArrayRef<uint8_t> FpmBytes;
  if (auto EC = FpmReader.readBytes(FpmBytes, FpmReader.bytesRemaining()))
    return EC;
  uint32_t BlocksRemaining = getBlockCount();
  uint32_t BI = 0;
  for (auto Byte : FpmBytes) {
    uint32_t BlocksThisByte = std::min(BlocksRemaining, 8U);
    for (uint32_t I = 0; I < BlocksThisByte; ++I) {
      if (Byte & (1 << I))
        ContainerLayout.FreePageMap[BI] = true;
      --BlocksRemaining;
      ++BI;
    }
  }

  Reader.setOffset(getBlockMapOffset());
  if (auto EC = Reader.readArray(ContainerLayout.DirectoryBlocks,
                                 getNumDirectoryBlocks()))
    return EC;

  return Error::success();
}

Error PDBFile::parseStreamData() {
#if LLVM_ENABLE_ZSTD
  if (MSFZ)
    return Error::success();
#endif
  assert(ContainerLayout.SB);
  if (DirectoryStream)
    return Error::success();

  uint32_t NumStreams = 0;

  // Normally you can't use a MappedBlockStream without having fully parsed the
  // PDB file, because it accesses the directory and various other things, which
  // is exactly what we are attempting to parse.  By specifying a custom
  // subclass of IPDBStreamData which only accesses the fields that have already
  // been parsed, we can avoid this and reuse MappedBlockStream.
  auto DS = MappedBlockStream::createDirectoryStream(ContainerLayout, *Buffer,
                                                     Allocator);
  BinaryStreamReader Reader(*DS);
  if (auto EC = Reader.readInteger(NumStreams))
    return EC;

  if (auto EC = Reader.readArray(ContainerLayout.StreamSizes, NumStreams))
    return EC;
  for (uint32_t I = 0; I < NumStreams; ++I) {
    uint32_t StreamSize = getStreamByteSize(I);
    // FIXME: What does StreamSize ~0U mean?
    uint64_t NumExpectedStreamBlocks =
        StreamSize == UINT32_MAX
            ? 0
            : msf::bytesToBlocks(StreamSize, ContainerLayout.SB->BlockSize);

    // For convenience, we store the block array contiguously.  This is because
    // if someone calls setStreamMap(), it is more convenient to be able to call
    // it with an ArrayRef instead of setting up a StreamRef.  Since the
    // DirectoryStream is cached in the class and thus lives for the life of the
    // class, we can be guaranteed that readArray() will return a stable
    // reference, even if it has to allocate from its internal pool.
    ArrayRef<support::ulittle32_t> Blocks;
    if (auto EC = Reader.readArray(Blocks, NumExpectedStreamBlocks))
      return EC;
    for (uint32_t Block : Blocks) {
      uint64_t BlockEndOffset =
          (uint64_t)(Block + 1) * ContainerLayout.SB->BlockSize;
      if (BlockEndOffset > getFileSize())
        return make_error<RawError>(raw_error_code::corrupt_file,
                                    "Stream block map is corrupt.");
    }
    ContainerLayout.StreamMap.push_back(Blocks);
  }

  // We should have read exactly SB->NumDirectoryBytes bytes.
  assert(Reader.bytesRemaining() == 0);
  DirectoryStream = std::move(DS);
  return Error::success();
}

ArrayRef<support::ulittle32_t> PDBFile::getDirectoryBlockArray() const {
  return ContainerLayout.DirectoryBlocks;
}

std::unique_ptr<MappedBlockStream>
PDBFile::createIndexedStream(uint16_t SN) const {
#if LLVM_ENABLE_ZSTD
  if (MSFZ)
    return nullptr;
#endif
  if (SN == kInvalidStreamIndex)
    return nullptr;
  return MappedBlockStream::createIndexedStream(ContainerLayout, *Buffer, SN,
                                                Allocator);
}

MSFStreamLayout PDBFile::getStreamLayout(uint32_t StreamIdx) const {
#if LLVM_ENABLE_ZSTD
  if (MSFZ)
    return {};
#endif
  MSFStreamLayout Result;
  auto Blocks = getStreamBlockList(StreamIdx);
  Result.Blocks.assign(Blocks.begin(), Blocks.end());
  Result.Length = getStreamByteSize(StreamIdx);
  return Result;
}

msf::MSFStreamLayout PDBFile::getFpmStreamLayout() const {
#if LLVM_ENABLE_ZSTD
  if (MSFZ)
    return {};
#endif
  return msf::getFpmStreamLayout(ContainerLayout);
}

bool PDBFile::isMSFZ() const {
#if LLVM_ENABLE_ZSTD
  return MSFZ != nullptr;
#else
  return false;
#endif
}

Expected<std::unique_ptr<BinaryStream>>
PDBFile::safelyCreateStream(uint32_t StreamIndex) const {
  if (StreamIndex == kInvalidStreamIndex || StreamIndex >= getNumStreams() ||
      getStreamByteSize(StreamIndex) == std::numeric_limits<uint32_t>::max())
    return make_error<RawError>(raw_error_code::no_stream);
#if LLVM_ENABLE_ZSTD
  if (MSFZ)
    return MSFZ->createStream(StreamIndex);
#endif
  return std::unique_ptr<BinaryStream>(MappedBlockStream::createIndexedStream(
      ContainerLayout, *Buffer, StreamIndex, Allocator));
}

Expected<GlobalsStream &> PDBFile::getPDBGlobalsStream() {
  if (!Globals) {
    auto DbiS = getPDBDbiStream();
    if (!DbiS)
      return DbiS.takeError();

    auto GlobalS = safelyCreateStream(DbiS->getGlobalSymbolStreamIndex());
    if (!GlobalS)
      return GlobalS.takeError();
    auto TempGlobals = std::make_unique<GlobalsStream>(std::move(*GlobalS));
    if (auto EC = TempGlobals->reload())
      return std::move(EC);
    Globals = std::move(TempGlobals);
  }
  return *Globals;
}

Expected<InfoStream &> PDBFile::getPDBInfoStream() {
  if (!Info) {
    auto InfoS = safelyCreateStream(StreamPDB);
    if (!InfoS)
      return InfoS.takeError();
    auto TempInfo = std::make_unique<InfoStream>(std::move(*InfoS));
    if (auto EC = TempInfo->reload())
      return std::move(EC);
    Info = std::move(TempInfo);
  }
  return *Info;
}

Expected<DbiStream &> PDBFile::getPDBDbiStream() {
  if (!Dbi) {
    auto DbiS = safelyCreateStream(StreamDBI);
    if (!DbiS)
      return DbiS.takeError();
    auto TempDbi = std::make_unique<DbiStream>(std::move(*DbiS));
    if (auto EC = TempDbi->reload(this))
      return std::move(EC);
    Dbi = std::move(TempDbi);
  }
  return *Dbi;
}

Expected<TpiStream &> PDBFile::getPDBTpiStream() {
  if (!Tpi) {
    auto TpiS = safelyCreateStream(StreamTPI);
    if (!TpiS)
      return TpiS.takeError();
    auto TempTpi = std::make_unique<TpiStream>(*this, std::move(*TpiS));
    if (auto EC = TempTpi->reload())
      return std::move(EC);
    Tpi = std::move(TempTpi);
  }
  return *Tpi;
}

Expected<TpiStream &> PDBFile::getPDBIpiStream() {
  if (!Ipi) {
    if (!hasPDBIpiStream())
      return make_error<RawError>(raw_error_code::no_stream);

    auto IpiS = safelyCreateStream(StreamIPI);
    if (!IpiS)
      return IpiS.takeError();
    auto TempIpi = std::make_unique<TpiStream>(*this, std::move(*IpiS));
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

    auto PublicS = safelyCreateStream(DbiS->getPublicSymbolStreamIndex());
    if (!PublicS)
      return PublicS.takeError();
    auto TempPublics = std::make_unique<PublicsStream>(std::move(*PublicS));
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
    auto SymbolS = safelyCreateStream(SymbolStreamNum);
    if (!SymbolS)
      return SymbolS.takeError();

    auto TempSymbols = std::make_unique<SymbolStream>(std::move(*SymbolS));
    if (auto EC = TempSymbols->reload())
      return std::move(EC);
    Symbols = std::move(TempSymbols);
  }
  return *Symbols;
}

Expected<PDBStringTable &> PDBFile::getStringTable() {
  if (!Strings) {
    auto NS = safelyCreateLogicalNamedStream("/names");
    if (!NS)
      return NS.takeError();

    auto N = std::make_unique<PDBStringTable>();
    BinaryStreamReader Reader(**NS);
    if (auto EC = N->reload(Reader))
      return std::move(EC);
    assert(Reader.bytesRemaining() == 0);
    StringTableStream = std::move(*NS);
    Strings = std::move(N);
  }
  return *Strings;
}

Expected<InjectedSourceStream &> PDBFile::getInjectedSourceStream() {
  if (!InjectedSources) {
    auto IJS = safelyCreateLogicalNamedStream("/src/headerblock");
    if (!IJS)
      return IJS.takeError();

    auto Strings = getStringTable();
    if (!Strings)
      return Strings.takeError();

    auto IJ = std::make_unique<InjectedSourceStream>(std::move(*IJS));
    if (auto EC = IJ->reload(*Strings))
      return std::move(EC);
    InjectedSources = std::move(IJ);
  }
  return *InjectedSources;
}

llvm::Expected<object::DXContainer &> PDBFile::getDXContainerStream() {
  if (!Dxc) {
    auto Stream = safelyCreateStream(StreamDXContainer);
    if (!Stream)
      return Stream.takeError();
    auto StreamSize = getStreamByteSize(StreamDXContainer);
    ArrayRef<uint8_t> StreamData;
    auto Error = Stream->get()->readBytes(0, StreamSize, StreamData);
    if (Error)
      return Error;

    StringRef Ref(reinterpret_cast<const char *>(StreamData.data()),
                  StreamSize);
    MemoryBufferRef MemBuf(Ref, "DXContainerStream");
    auto DXC = object::DXContainer::create(MemBuf);
    if (!DXC)
      return DXC.takeError();
    Dxc = std::make_unique<object::DXContainer>(std::move(*DXC));
  }
  return *Dxc;
}

uint32_t PDBFile::getPointerSize() {
  auto DbiS = getPDBDbiStream();
  if (!DbiS)
    return 0;
  PDB_Machine Machine = DbiS->getMachineType();
  if (Machine == PDB_Machine::Amd64)
    return 8;
  return 4;
}

bool PDBFile::hasPDBDbiStream() const {
  return StreamDBI < getNumStreams() && getStreamByteSize(StreamDBI) > 0;
}

bool PDBFile::hasPDBGlobalsStream() {
  auto DbiS = getPDBDbiStream();
  if (!DbiS) {
    consumeError(DbiS.takeError());
    return false;
  }

  return DbiS->getGlobalSymbolStreamIndex() < getNumStreams();
}

bool PDBFile::hasPDBInfoStream() const { return StreamPDB < getNumStreams(); }

bool PDBFile::hasPDBIpiStream() const {
  if (!hasPDBInfoStream())
    return false;

  if (StreamIPI >= getNumStreams())
    return false;

  auto &InfoStream = cantFail(const_cast<PDBFile *>(this)->getPDBInfoStream());
  return InfoStream.containsIdStream();
}

bool PDBFile::hasPDBPublicsStream() {
  auto DbiS = getPDBDbiStream();
  if (!DbiS) {
    consumeError(DbiS.takeError());
    return false;
  }
  return DbiS->getPublicSymbolStreamIndex() < getNumStreams();
}

bool PDBFile::hasPDBSymbolStream() {
  auto DbiS = getPDBDbiStream();
  if (!DbiS)
    return false;
  return DbiS->getSymRecordStreamIndex() < getNumStreams();
}

bool PDBFile::hasPDBTpiStream() const {
  return StreamTPI < getNumStreams() && getStreamByteSize(StreamTPI) != 0;
}

bool PDBFile::hasPDBStringTable() {
  auto IS = getPDBInfoStream();
  if (!IS)
    return false;
  Expected<uint32_t> ExpectedNSI = IS->getNamedStreamIndex("/names");
  if (!ExpectedNSI) {
    consumeError(ExpectedNSI.takeError());
    return false;
  }
  assert(*ExpectedNSI < getNumStreams());
  return true;
}

bool PDBFile::hasPDBInjectedSourceStream() {
  auto IS = getPDBInfoStream();
  if (!IS)
    return false;
  Expected<uint32_t> ExpectedNSI = IS->getNamedStreamIndex("/src/headerblock");
  if (!ExpectedNSI) {
    consumeError(ExpectedNSI.takeError());
    return false;
  }
  assert(*ExpectedNSI < getNumStreams());
  return true;
}

/// Wrapper around MappedBlockStream::createIndexedStream() that checks if a
/// stream with that index actually exists.  If it does not, the return value
/// will have an MSFError with code msf_error_code::no_stream.  Else, the return
/// value will contain the stream returned by createIndexedStream().
Expected<std::unique_ptr<MappedBlockStream>>
PDBFile::safelyCreateIndexedStream(uint32_t StreamIndex) const {
#if LLVM_ENABLE_ZSTD
  if (MSFZ)
    return make_error<RawError>(
        raw_error_code::feature_unsupported,
        "MSFZ files do not have MSF block stream layouts");
#endif
  if (StreamIndex >= getNumStreams())
    // This rejects kInvalidStreamIndex with an error as well.
    return make_error<RawError>(raw_error_code::no_stream);
  return createIndexedStream(StreamIndex);
}

Expected<std::unique_ptr<MappedBlockStream>>
PDBFile::safelyCreateNamedStream(StringRef Name) {
  auto IS = getPDBInfoStream();
  if (!IS)
    return IS.takeError();

  Expected<uint32_t> ExpectedNSI = IS->getNamedStreamIndex(Name);
  if (!ExpectedNSI)
    return ExpectedNSI.takeError();
  uint32_t NameStreamIndex = *ExpectedNSI;

  return safelyCreateIndexedStream(NameStreamIndex);
}

Expected<std::unique_ptr<BinaryStream>>
PDBFile::safelyCreateLogicalNamedStream(StringRef Name) {
  auto IS = getPDBInfoStream();
  if (!IS)
    return IS.takeError();

  Expected<uint32_t> ExpectedNSI = IS->getNamedStreamIndex(Name);
  if (!ExpectedNSI)
    return ExpectedNSI.takeError();
  return safelyCreateStream(*ExpectedNSI);
}
