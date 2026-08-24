//===- MSFZTest.cpp - Tests for the MSFZ container ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/MSF/MSFZ.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/DebugInfo/MSF/MSFCommon.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/BinaryByteStream.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>

using namespace llvm;
using namespace llvm::msf;
using namespace testing;

static void writeU32(MutableArrayRef<uint8_t> Data, uint64_t Offset,
                     uint32_t Value) {
  support::ulittle32_t LE;
  LE = Value;
  std::memcpy(Data.data() + Offset, &LE, sizeof(LE));
}

static void writeU64(MutableArrayRef<uint8_t> Data, uint64_t Offset,
                     uint64_t Value) {
  support::ulittle64_t LE;
  LE = Value;
  std::memcpy(Data.data() + Offset, &LE, sizeof(LE));
}

static SmallVector<uint8_t, 0> makeRawDeflateFile() {
  // These streams were produced independently with Python's
  // zlib.compressobj(wbits=-15).
  const uint8_t Directory[] = {0x63, 0x60, 0x60, 0x60, 0x60, 0x63,
                               0x80, 0x83, 0x06, 0x10, 0x01, 0x00};
  const uint8_t FirstChunk[] = {0x4b, 0x4c, 0x4a, 0x06, 0x00};
  const uint8_t SecondChunk[] = {0x4b, 0x49, 0x4d, 0x03, 0x00};

  uint64_t DirectoryOffset = sizeof(MSFZHeader);
  uint64_t FirstChunkOffset = DirectoryOffset + sizeof(Directory);
  uint64_t SecondChunkOffset = FirstChunkOffset + sizeof(FirstChunk);
  uint64_t ChunkTableOffset = SecondChunkOffset + sizeof(SecondChunk);
  SmallVector<uint8_t, 0> File(ChunkTableOffset + 2 * sizeof(MSFZChunkEntry));

  MSFZHeader Header = {};
  std::memcpy(Header.Magic, MSFZMagic, sizeof(MSFZMagic));
  Header.StreamDirectoryOffset = DirectoryOffset;
  Header.StreamDirectoryCompression =
      static_cast<uint32_t>(MSFZCompression::Deflate);
  Header.StreamDirectorySizeCompressed = sizeof(Directory);
  Header.StreamDirectorySizeUncompressed = 20;
  Header.NumStreams = 2;
  Header.ChunkTableOffset = ChunkTableOffset;
  Header.NumChunks = 2;
  Header.ChunkTableSize = 2 * sizeof(MSFZChunkEntry);
  std::memcpy(File.data(), &Header, sizeof(Header));
  std::memcpy(File.data() + DirectoryOffset, Directory, sizeof(Directory));
  std::memcpy(File.data() + FirstChunkOffset, FirstChunk, sizeof(FirstChunk));
  std::memcpy(File.data() + SecondChunkOffset, SecondChunk,
              sizeof(SecondChunk));

  MSFZChunkEntry Chunks[2] = {};
  Chunks[0].FileOffset = FirstChunkOffset;
  Chunks[0].Compression = static_cast<uint32_t>(MSFZCompression::Deflate);
  Chunks[0].CompressedSize = sizeof(FirstChunk);
  Chunks[0].UncompressedSize = 3;
  Chunks[1].FileOffset = SecondChunkOffset;
  Chunks[1].Compression = static_cast<uint32_t>(MSFZCompression::Deflate);
  Chunks[1].CompressedSize = sizeof(SecondChunk);
  Chunks[1].UncompressedSize = 3;
  std::memcpy(File.data() + ChunkTableOffset, Chunks, sizeof(Chunks));
  return File;
}

TEST(MSFZTest, ReadsRawDeflateDirectoryAndChunks) {
  SmallVector<uint8_t, 0> File = makeRawDeflateFile();
  BumpPtrAllocator Allocator;
  BinaryByteStream Input(File, llvm::endianness::little);
  auto ReaderOrErr = MSFZFile::create(Input, Allocator);
#if LLVM_ENABLE_ZLIB
  ASSERT_THAT_EXPECTED(ReaderOrErr, Succeeded());
  auto StreamOrErr = (*ReaderOrErr)->createStream(1);
  ASSERT_THAT_EXPECTED(StreamOrErr, Succeeded());
  ArrayRef<uint8_t> Data;
  ASSERT_THAT_ERROR((*StreamOrErr)->readBytes(2, 3, Data), Succeeded());
  EXPECT_EQ(toStringRef(Data), "cde");
#else
  EXPECT_THAT_EXPECTED(
      ReaderOrErr,
      FailedWithMessage(
          "MSFZ raw DEFLATE compression requires LLVM_ENABLE_ZLIB"));

  MSFZHeader Header;
  std::memcpy(&Header, File.data(), sizeof(Header));
  Header.NumChunks = 0;
  Header.ChunkTableSize = 0;
  std::memcpy(File.data(), &Header, sizeof(Header));
  BinaryByteStream DirectoryInput(File, llvm::endianness::little);
  EXPECT_THAT_EXPECTED(
      MSFZFile::create(DirectoryInput, Allocator),
      FailedWithMessage(
          "MSFZ raw DEFLATE directory compression requires LLVM_ENABLE_ZLIB"));
#endif
}

TEST(MSFZTest, CompatibilityWrapperWritesCompressedStreams) {
  constexpr uint32_t BlockSize = 512;
  constexpr StringLiteral Contents = "hello MSFZ";

  SuperBlock SB = {};
  SB.BlockSize = BlockSize;
  SB.NumBlocks = 2;

  support::ulittle32_t Sizes[3];
  Sizes[0] = 0;
  Sizes[1] = Contents.size();
  Sizes[2] = std::numeric_limits<uint32_t>::max();
  support::ulittle32_t StreamBlock;
  StreamBlock = 1;

  MSFLayout Layout;
  Layout.SB = &SB;
  Layout.StreamSizes = Sizes;
  Layout.StreamMap = {{}, {&StreamBlock, 1}, {}};

  SmallVector<uint8_t, BlockSize * 2> MsfData(BlockSize * 2);
  std::memcpy(MsfData.data() + BlockSize, Contents.data(), Contents.size());

  unittest::TempDir TestDirectory("msfz-test", /*Unique=*/true);
  SmallString<128> Path = TestDirectory.path("test.pdz");
  ASSERT_THAT_ERROR(writeMSFZ(Path, Layout, MsfData), Succeeded());

  auto FileOrErr = MemoryBuffer::getFile(Path, /*IsText=*/false);
  ASSERT_TRUE(FileOrErr);
  ArrayRef<uint8_t> File(
      reinterpret_cast<const uint8_t *>((*FileOrErr)->getBufferStart()),
      (*FileOrErr)->getBufferSize());
  ASSERT_GE(File.size(), sizeof(MSFZHeader));

  MSFZHeader Header = {};
  std::memcpy(&Header, File.data(), sizeof(Header));
  EXPECT_EQ(ArrayRef(Header.Magic), ArrayRef(MSFZMagic));
  EXPECT_EQ(Header.Version, 0u);
  EXPECT_EQ(Header.NumStreams, 3u);
  EXPECT_EQ(Header.NumChunks, 1u);
  EXPECT_EQ(Header.StreamDirectoryCompression,
            static_cast<uint32_t>(MSFZCompression::None));
  EXPECT_EQ(Header.ChunkTableSize, sizeof(MSFZChunkEntry));

  ArrayRef<uint8_t> Directory = File.slice(
      Header.StreamDirectoryOffset, Header.StreamDirectorySizeCompressed);
  EXPECT_EQ(Header.StreamDirectorySizeCompressed,
            Header.StreamDirectorySizeUncompressed);

  const uint8_t ExpectedDirectory[] = {
      0, 0, 0, 0,   10, 0, 0, 0, 0,    0,    0,    0,
      0, 0, 0, 128, 0,  0, 0, 0, 0xff, 0xff, 0xff, 0xff,
  };
  EXPECT_EQ(Directory, ArrayRef(ExpectedDirectory));

  MSFZChunkEntry Chunk = {};
  std::memcpy(&Chunk, File.data() + Header.ChunkTableOffset, sizeof(Chunk));
  EXPECT_EQ(Chunk.Compression, static_cast<uint32_t>(MSFZCompression::Zstd));
  EXPECT_EQ(Chunk.UncompressedSize, Contents.size());

  SmallVector<uint8_t, 16> StreamData;
  ASSERT_THAT_ERROR(compression::zstd::decompress(
                        File.slice(Chunk.FileOffset, Chunk.CompressedSize),
                        StreamData, Chunk.UncompressedSize),
                    Succeeded());
  EXPECT_EQ(StreamData,
            ArrayRef(reinterpret_cast<const uint8_t *>(Contents.data()),
                     Contents.size()));

  BumpPtrAllocator Allocator;
  BinaryByteStream Input(File, llvm::endianness::little);
  auto ReaderOrErr = MSFZFile::create(Input, Allocator);
  ASSERT_THAT_EXPECTED(ReaderOrErr, Succeeded());
  ASSERT_EQ((*ReaderOrErr)->getNumStreams(), 3u);
  EXPECT_TRUE((*ReaderOrErr)->isNilStream(2));

  auto StreamOrErr = (*ReaderOrErr)->createStream(1);
  ASSERT_THAT_EXPECTED(StreamOrErr, Succeeded());
  ArrayRef<uint8_t> ReadData;
  ASSERT_THAT_ERROR((*StreamOrErr)->readBytes(0, Contents.size(), ReadData),
                    Succeeded());
  EXPECT_EQ(ReadData,
            ArrayRef(reinterpret_cast<const uint8_t *>(Contents.data()),
                     Contents.size()));
}

TEST(MSFZTest, SplitsFragmentsAtChunkBoundaries) {
  constexpr uint32_t TestChunkSize = 1024 * 1024;
  constexpr uint32_t BlockSize = TestChunkSize + 1;
  SuperBlock SB = {};
  SB.BlockSize = BlockSize;
  SB.NumBlocks = 2;

  support::ulittle32_t Sizes[2];
  Sizes[0] = 0;
  Sizes[1] = BlockSize;
  support::ulittle32_t StreamBlock;
  StreamBlock = 1;

  MSFLayout Layout;
  Layout.SB = &SB;
  Layout.StreamSizes = Sizes;
  Layout.StreamMap = {{}, {&StreamBlock, 1}};

  SmallVector<uint8_t, 0> MsfData(BlockSize * 2);
  MsfData[BlockSize + TestChunkSize - 1] = 'A';
  MsfData[BlockSize + TestChunkSize] = 'B';
  unittest::TempDir TestDirectory("msfz-chunks-test", /*Unique=*/true);
  SmallString<128> Path = TestDirectory.path("test.pdz");
  ASSERT_THAT_ERROR(writeMSFZ(Path, Layout, MsfData), Succeeded());

  auto FileOrErr = MemoryBuffer::getFile(Path, /*IsText=*/false);
  ASSERT_TRUE(FileOrErr);
  ArrayRef<uint8_t> File(
      reinterpret_cast<const uint8_t *>((*FileOrErr)->getBufferStart()),
      (*FileOrErr)->getBufferSize());
  MSFZHeader Header = {};
  std::memcpy(&Header, File.data(), sizeof(Header));
  EXPECT_EQ(Header.NumChunks, 2u);

  ArrayRef<uint8_t> Directory = File.slice(
      Header.StreamDirectoryOffset, Header.StreamDirectorySizeUncompressed);
  const uint8_t ExpectedDirectory[] = {
      0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0,   0, 0, 0, 128,
      1, 0, 0, 0, 0, 0, 0,  0, 1, 0, 0, 128, 0, 0, 0, 0,
  };
  EXPECT_EQ(Directory, ArrayRef(ExpectedDirectory));

  SmallVector<uint8_t, 0> CrossChunkFile(File.begin(), File.end());
  // Replace the generated split fragments with one logical fragment spanning
  // both chunks to exercise cross-chunk reads.
  MSFZHeader CrossChunkHeader;
  std::memcpy(&CrossChunkHeader, CrossChunkFile.data(),
              sizeof(CrossChunkHeader));
  CrossChunkHeader.StreamDirectorySizeCompressed = 20;
  CrossChunkHeader.StreamDirectorySizeUncompressed = 20;
  std::memcpy(CrossChunkFile.data(), &CrossChunkHeader,
              sizeof(CrossChunkHeader));
  uint64_t StreamDirectoryOffset = CrossChunkHeader.StreamDirectoryOffset;
  writeU32(CrossChunkFile, StreamDirectoryOffset + 4, BlockSize);
  writeU32(CrossChunkFile, StreamDirectoryOffset + 16, 0);

  BumpPtrAllocator Allocator;
  BinaryByteStream Input(CrossChunkFile, llvm::endianness::little);
  auto ReaderOrErr = MSFZFile::create(Input, Allocator);
  ASSERT_THAT_EXPECTED(ReaderOrErr, Succeeded());
  auto StreamOrErr = (*ReaderOrErr)->createStream(1);
  ASSERT_THAT_EXPECTED(StreamOrErr, Succeeded());
  ArrayRef<uint8_t> ReadData;
  ASSERT_THAT_ERROR((*StreamOrErr)->readBytes(TestChunkSize - 1, 2, ReadData),
                    Succeeded());
  const uint8_t ExpectedData[] = {'A', 'B'};
  EXPECT_EQ(ReadData, ArrayRef(ExpectedData));
}

TEST(MSFZTest, ParallelCompressionIsDeterministic) {
  ThreadPoolStrategy SavedStrategy = parallel::strategy;
  scope_exit RestoreStrategy([&] { parallel::strategy = SavedStrategy; });

  constexpr uint32_t ChunkSize = 1024 * 1024;
  constexpr uint32_t StreamSize = 6 * ChunkSize + 123;
  SuperBlock SB = {};
  SB.BlockSize = StreamSize;
  SB.NumBlocks = 2;

  support::ulittle32_t Sizes[2];
  Sizes[0] = 0;
  Sizes[1] = StreamSize;
  support::ulittle32_t StreamBlock;
  StreamBlock = 1;

  MSFLayout Layout;
  Layout.SB = &SB;
  Layout.StreamSizes = Sizes;
  Layout.StreamMap = {{}, {&StreamBlock, 1}};

  SmallVector<uint8_t, 0> MsfData(uint64_t(StreamSize) * 2);
  for (uint32_t I = 0; I != StreamSize; ++I)
    MsfData[StreamSize + I] = static_cast<uint8_t>(I * 37);

  unittest::TempDir TestDirectory("msfz-parallel-test", /*Unique=*/true);
  SmallString<128> SerialPath = TestDirectory.path("serial.pdz");
  parallel::strategy = hardware_concurrency(1);
  ASSERT_THAT_ERROR(writeMSFZ(SerialPath, Layout, MsfData), Succeeded());

  SmallString<128> ParallelPath = TestDirectory.path("parallel.pdz");
  parallel::strategy = hardware_concurrency(4);
  ASSERT_THAT_ERROR(writeMSFZ(ParallelPath, Layout, MsfData), Succeeded());

  auto SerialOrErr = MemoryBuffer::getFile(SerialPath, /*IsText=*/false);
  ASSERT_TRUE(SerialOrErr);
  auto ParallelOrErr = MemoryBuffer::getFile(ParallelPath, /*IsText=*/false);
  ASSERT_TRUE(ParallelOrErr);
  EXPECT_EQ((*SerialOrErr)->getBuffer(), (*ParallelOrErr)->getBuffer());
}

TEST(MSFZTest, ReadsUncompressedFragmentsAndRejectsMalformedMetadata) {
  SmallVector<uint8_t, 0> File(sizeof(MSFZHeader) + 6 + 32 +
                               sizeof(MSFZChunkEntry));
  MSFZHeader Header = {};
  std::memcpy(Header.Magic, MSFZMagic, sizeof(MSFZMagic));
  Header.Version = 0;
  Header.StreamDirectoryOffset = sizeof(MSFZHeader) + 6;
  Header.StreamDirectoryCompression =
      static_cast<uint32_t>(MSFZCompression::None);
  Header.StreamDirectorySizeCompressed = 32;
  Header.StreamDirectorySizeUncompressed = 32;
  Header.NumStreams = 2;
  Header.ChunkTableOffset = Header.StreamDirectoryOffset + 32;
  Header.NumChunks = 1;
  Header.ChunkTableSize = sizeof(MSFZChunkEntry);
  std::memcpy(File.data(), &Header, sizeof(Header));
  std::memcpy(File.data() + sizeof(Header), "rawzip", 6);

  uint64_t DirectoryOffset = Header.StreamDirectoryOffset;
  writeU32(File, DirectoryOffset, 0);
  writeU32(File, DirectoryOffset + 4, 3);
  writeU64(File, DirectoryOffset + 8, sizeof(MSFZHeader));
  writeU32(File, DirectoryOffset + 16, 3);
  writeU64(File, DirectoryOffset + 20, uint64_t{1} << 63);
  writeU32(File, DirectoryOffset + 28, 0);

  MSFZChunkEntry Chunk = {};
  Chunk.FileOffset = sizeof(MSFZHeader) + 3;
  Chunk.Compression = static_cast<uint32_t>(MSFZCompression::None);
  Chunk.CompressedSize = 3;
  Chunk.UncompressedSize = 3;
  std::memcpy(File.data() + Header.ChunkTableOffset, &Chunk, sizeof(Chunk));

  BumpPtrAllocator Allocator;
  BinaryByteStream Input(File, llvm::endianness::little);
  auto ReaderOrErr = MSFZFile::create(Input, Allocator);
  ASSERT_THAT_EXPECTED(ReaderOrErr, Succeeded());
  auto StreamOrErr = (*ReaderOrErr)->createStream(1);
  ASSERT_THAT_EXPECTED(StreamOrErr, Succeeded());
  ArrayRef<uint8_t> Data;
  ASSERT_THAT_ERROR((*StreamOrErr)->readBytes(0, 6, Data), Succeeded());
  const uint8_t ExpectedData[] = {'r', 'a', 'w', 'z', 'i', 'p'};
  EXPECT_EQ(Data, ArrayRef(ExpectedData));

  Header.ChunkTableSize = 0;
  std::memcpy(File.data(), &Header, sizeof(Header));
  BinaryByteStream MalformedInput(File, llvm::endianness::little);
  EXPECT_THAT_EXPECTED(MSFZFile::create(MalformedInput, Allocator), Failed());
}

TEST(MSFZTest, RejectsInvalidPhysicalLayout) {
  constexpr uint64_t UncompressedOffset = sizeof(MSFZHeader);
  constexpr uint64_t ChunkOffset = UncompressedOffset + 3;
  constexpr uint64_t DirectoryOffset = ChunkOffset + 3;
  constexpr uint64_t ChunkTableOffset = DirectoryOffset + 16;
  SmallVector<uint8_t, 0> File(ChunkTableOffset + sizeof(MSFZChunkEntry));

  MSFZHeader Header = {};
  std::memcpy(Header.Magic, MSFZMagic, sizeof(MSFZMagic));
  Header.StreamDirectoryOffset = DirectoryOffset;
  Header.StreamDirectoryCompression =
      static_cast<uint32_t>(MSFZCompression::None);
  Header.StreamDirectorySizeCompressed = 16;
  Header.StreamDirectorySizeUncompressed = 16;
  Header.NumStreams = 1;
  Header.ChunkTableOffset = ChunkTableOffset;
  Header.NumChunks = 1;
  Header.ChunkTableSize = sizeof(MSFZChunkEntry);
  std::memcpy(File.data(), &Header, sizeof(Header));
  std::memcpy(File.data() + UncompressedOffset, "rawzip", 6);

  writeU32(File, DirectoryOffset, 3);
  writeU64(File, DirectoryOffset + 4, UncompressedOffset);
  writeU32(File, DirectoryOffset + 12, 0);

  MSFZChunkEntry Chunk = {};
  Chunk.FileOffset = ChunkOffset;
  Chunk.Compression = static_cast<uint32_t>(MSFZCompression::None);
  Chunk.CompressedSize = 3;
  Chunk.UncompressedSize = 3;
  std::memcpy(File.data() + ChunkTableOffset, &Chunk, sizeof(Chunk));

  BumpPtrAllocator Allocator;
  BinaryByteStream ValidInput(File, llvm::endianness::little);
  ASSERT_THAT_EXPECTED(MSFZFile::create(ValidInput, Allocator), Succeeded());

  SmallVector<uint8_t, 0> ReservedBitsFile = File;
  writeU64(ReservedBitsFile, DirectoryOffset + 4,
           UncompressedOffset | (uint64_t{1} << 48));
  BinaryByteStream ReservedBitsInput(ReservedBitsFile,
                                     llvm::endianness::little);
  EXPECT_THAT_EXPECTED(
      MSFZFile::create(ReservedBitsInput, Allocator),
      FailedWithMessage(
          "MSFZ uncompressed fragment has non-zero reserved bits"));

  SmallVector<uint8_t, 0> ZeroSizedChunkFile = File;
  writeU32(ZeroSizedChunkFile,
           ChunkTableOffset + offsetof(MSFZChunkEntry, CompressedSize), 0);
  BinaryByteStream ZeroSizedChunkInput(ZeroSizedChunkFile,
                                       llvm::endianness::little);
  EXPECT_THAT_EXPECTED(MSFZFile::create(ZeroSizedChunkInput, Allocator),
                       FailedWithMessage("MSFZ file has a zero-sized chunk"));

  SmallVector<uint8_t, 0> AliasedPayloadFile = File;
  writeU64(AliasedPayloadFile,
           ChunkTableOffset + offsetof(MSFZChunkEntry, FileOffset),
           UncompressedOffset);
  BinaryByteStream AliasedPayloadInput(AliasedPayloadFile,
                                       llvm::endianness::little);
  EXPECT_THAT_EXPECTED(
      MSFZFile::create(AliasedPayloadInput, Allocator),
      FailedWithMessage("MSFZ file has overlapping physical regions"));

  SmallVector<uint8_t, 0> MetadataOverlapFile = File;
  writeU64(MetadataOverlapFile,
           ChunkTableOffset + offsetof(MSFZChunkEntry, FileOffset),
           ChunkTableOffset);
  BinaryByteStream MetadataOverlapInput(MetadataOverlapFile,
                                        llvm::endianness::little);
  EXPECT_THAT_EXPECTED(
      MSFZFile::create(MetadataOverlapInput, Allocator),
      FailedWithMessage("MSFZ file has overlapping physical regions"));
}

TEST(MSFZTest, RejectsUnboundedMetadataAllocations) {
  SmallVector<uint8_t, 0> File(sizeof(MSFZHeader) + 4);
  MSFZHeader Header = {};
  std::memcpy(Header.Magic, MSFZMagic, sizeof(MSFZMagic));
  Header.Version = 0;
  Header.StreamDirectoryOffset = sizeof(MSFZHeader);
  Header.StreamDirectoryCompression =
      static_cast<uint32_t>(MSFZCompression::None);
  Header.StreamDirectorySizeCompressed = 4;
  Header.StreamDirectorySizeUncompressed = 4;
  Header.NumStreams = std::numeric_limits<uint32_t>::max();
  Header.ChunkTableOffset = File.size();
  std::memcpy(File.data(), &Header, sizeof(Header));

  BumpPtrAllocator Allocator;
  BinaryByteStream TooManyStreams(File, llvm::endianness::little);
  EXPECT_THAT_EXPECTED(MSFZFile::create(TooManyStreams, Allocator), Failed());

  Header.NumStreams = 1;
  Header.StreamDirectoryCompression =
      static_cast<uint32_t>(MSFZCompression::Zstd);
  Header.StreamDirectorySizeUncompressed = std::numeric_limits<uint32_t>::max();
  std::memcpy(File.data(), &Header, sizeof(Header));
  BinaryByteStream OversizedDirectory(File, llvm::endianness::little);
  EXPECT_THAT_EXPECTED(MSFZFile::create(OversizedDirectory, Allocator),
                       Failed());

  File.resize(sizeof(MSFZHeader) + 4 + sizeof(MSFZChunkEntry));
  Header.StreamDirectoryCompression =
      static_cast<uint32_t>(MSFZCompression::None);
  Header.StreamDirectorySizeUncompressed = 4;
  Header.ChunkTableOffset = sizeof(MSFZHeader) + 4;
  Header.NumChunks = 1;
  Header.ChunkTableSize = sizeof(MSFZChunkEntry);
  std::memcpy(File.data(), &Header, sizeof(Header));
  MSFZChunkEntry Chunk = {};
  Chunk.FileOffset = sizeof(MSFZHeader);
  Chunk.Compression = static_cast<uint32_t>(MSFZCompression::Zstd);
  Chunk.CompressedSize = 4;
  Chunk.UncompressedSize = std::numeric_limits<uint32_t>::max();
  std::memcpy(File.data() + Header.ChunkTableOffset, &Chunk, sizeof(Chunk));
  BinaryByteStream OversizedChunk(File, llvm::endianness::little);
  EXPECT_THAT_EXPECTED(MSFZFile::create(OversizedChunk, Allocator), Failed());
}

TEST(MSFZTest, StreamsShuffledPhysicalBlocks) {
  constexpr uint32_t BlockSize = 512;
  SuperBlock SB = {};
  SB.BlockSize = BlockSize;
  SB.NumBlocks = 5;

  support::ulittle32_t Sizes[2];
  Sizes[0] = 0;
  Sizes[1] = BlockSize + 7;
  support::ulittle32_t StreamBlocks[2];
  StreamBlocks[0] = 3;
  StreamBlocks[1] = 1;

  MSFLayout Layout;
  Layout.SB = &SB;
  Layout.StreamSizes = Sizes;
  Layout.StreamMap = {{}, StreamBlocks};

  unittest::TempDir TestDirectory("msfz-streaming-test", /*Unique=*/true);
  SmallString<128> Path = TestDirectory.path("test.pdz");
  auto WriterOrErr =
      MSFZWriter::create(Path, Layout, compression::zstd::BestSpeedCompression);
  ASSERT_THAT_EXPECTED(WriterOrErr, Succeeded());
  std::unique_ptr<MSFZWriter> Writer = std::move(*WriterOrErr);

  SmallVector<uint8_t, 0> FirstBlock(BlockSize, 'A');
  SmallVector<uint8_t, 0> SecondBlock(7, 'B');
  ASSERT_THAT_ERROR(Writer->writeBytes(3 * BlockSize, FirstBlock), Succeeded());
  ASSERT_THAT_ERROR(Writer->writeBytes(BlockSize, SecondBlock), Succeeded());
  ASSERT_THAT_ERROR(Writer->finalize(), Succeeded());

  auto FileOrErr = MemoryBuffer::getFile(Path, /*IsText=*/false);
  ASSERT_TRUE(FileOrErr);
  ArrayRef<uint8_t> File(
      reinterpret_cast<const uint8_t *>((*FileOrErr)->getBufferStart()),
      (*FileOrErr)->getBufferSize());
  MSFZHeader Header = {};
  std::memcpy(&Header, File.data(), sizeof(Header));
  ASSERT_EQ(Header.NumChunks, 1u);
  MSFZChunkEntry Chunk = {};
  std::memcpy(&Chunk, File.data() + Header.ChunkTableOffset, sizeof(Chunk));

  SmallVector<uint8_t, 0> StreamData;
  ASSERT_THAT_ERROR(compression::zstd::decompress(
                        File.slice(Chunk.FileOffset, Chunk.CompressedSize),
                        StreamData, Chunk.UncompressedSize),
                    Succeeded());
  SmallVector<uint8_t, 0> Expected;
  append_range(Expected, FirstBlock);
  append_range(Expected, SecondBlock);
  EXPECT_EQ(StreamData, Expected);
}

TEST(MSFZTest, CanonicalDigestIgnoresWriteScheduling) {
  constexpr uint32_t BlockSize = 512;
  SuperBlock SB = {};
  SB.BlockSize = BlockSize;
  SB.NumBlocks = 3;

  support::ulittle32_t Sizes[3];
  Sizes[0] = 0;
  Sizes[1] = 2;
  Sizes[2] = 2;
  support::ulittle32_t FirstStreamBlock;
  FirstStreamBlock = 1;
  support::ulittle32_t SecondStreamBlock;
  SecondStreamBlock = 2;

  MSFLayout Layout;
  Layout.SB = &SB;
  Layout.StreamSizes = Sizes;
  Layout.StreamMap = {{}, {&FirstStreamBlock, 1}, {&SecondStreamBlock, 1}};

  unittest::TempDir TestDirectory("msfz-digest-test", /*Unique=*/true);
  SmallString<128> FirstPath = TestDirectory.path("first.pdz");
  auto FirstWriterOrErr = MSFZWriter::create(
      FirstPath, Layout, compression::zstd::BestSpeedCompression);
  ASSERT_THAT_EXPECTED(FirstWriterOrErr, Succeeded());
  std::unique_ptr<MSFZWriter> FirstWriter = std::move(*FirstWriterOrErr);

  const uint8_t FirstData[] = {'a', 'b'};
  const uint8_t SecondData[] = {'c', 'd'};
  ASSERT_THAT_ERROR(FirstWriter->writeBytes(BlockSize, FirstData), Succeeded());
  ASSERT_THAT_ERROR(FirstWriter->writeBytes(2 * BlockSize, SecondData),
                    Succeeded());
  auto FirstDigest = FirstWriter->getCanonicalDigest();
  ASSERT_THAT_EXPECTED(FirstDigest, Succeeded());
  ASSERT_THAT_ERROR(FirstWriter->finalize(), Succeeded());

  SmallString<128> SecondPath = TestDirectory.path("second.pdz");
  auto SecondWriterOrErr = MSFZWriter::create(
      SecondPath, Layout, compression::zstd::BestSpeedCompression);
  ASSERT_THAT_EXPECTED(SecondWriterOrErr, Succeeded());
  std::unique_ptr<MSFZWriter> SecondWriter = std::move(*SecondWriterOrErr);
  ASSERT_THAT_ERROR(SecondWriter->writeBytes(2 * BlockSize, SecondData),
                    Succeeded());
  ASSERT_THAT_ERROR(SecondWriter->writeBytes(BlockSize, FirstData),
                    Succeeded());
  auto SecondDigest = SecondWriter->getCanonicalDigest();
  ASSERT_THAT_EXPECTED(SecondDigest, Succeeded());
  ASSERT_THAT_ERROR(SecondWriter->finalize(), Succeeded());

  EXPECT_EQ(*FirstDigest, *SecondDigest);
}

TEST(MSFZTest, InitialUncompressedStreamIsPatchable) {
  ThreadPoolStrategy SavedStrategy = parallel::strategy;
  scope_exit RestoreStrategy([&] { parallel::strategy = SavedStrategy; });

  constexpr uint32_t BlockSize = 512;
  SuperBlock SB = {};
  SB.BlockSize = BlockSize;
  SB.NumBlocks = 2;

  support::ulittle32_t Sizes[2];
  Sizes[0] = 0;
  Sizes[1] = BlockSize;
  support::ulittle32_t StreamBlock;
  StreamBlock = 1;

  MSFLayout Layout;
  Layout.SB = &SB;
  Layout.StreamSizes = Sizes;
  Layout.StreamMap = {{}, {&StreamBlock, 1}};

  unittest::TempDir TestDirectory("msfz-patch-test", /*Unique=*/true);
  SmallVector<uint8_t, 0> Data(BlockSize);
  for (uint32_t I = 0; I != Data.size(); ++I)
    Data[I] = I;
  const uint8_t Patch[] = {4, 3, 2, 1};

  auto Write = [&](StringRef Path, unsigned ThreadCount) -> Error {
    parallel::strategy = hardware_concurrency(ThreadCount);
    auto WriterOrErr = MSFZWriter::create(
        Path, Layout, compression::zstd::BestSpeedCompression);
    if (!WriterOrErr)
      return WriterOrErr.takeError();
    std::unique_ptr<MSFZWriter> Writer = std::move(*WriterOrErr);
    if (Error E = Writer->setInitialUncompressedStream(
            /*StreamIndex=*/1, sizeof(Patch)))
      return E;
    if (Error E = Writer->writeBytes(BlockSize, Data))
      return E;
    if (Error E = Writer->patchStream(/*StreamIndex=*/1, /*Offset=*/0, Patch))
      return E;
    return Writer->finalize();
  };

  SmallString<128> SerialPath = TestDirectory.path("serial.pdz");
  ASSERT_THAT_ERROR(Write(SerialPath, 1), Succeeded());
  SmallString<128> ParallelPath = TestDirectory.path("parallel.pdz");
  ASSERT_THAT_ERROR(Write(ParallelPath, 4), Succeeded());

  auto SerialOrErr = MemoryBuffer::getFile(SerialPath, /*IsText=*/false);
  ASSERT_TRUE(SerialOrErr);
  auto FileOrErr = MemoryBuffer::getFile(ParallelPath, /*IsText=*/false);
  ASSERT_TRUE(FileOrErr);
  EXPECT_EQ((*SerialOrErr)->getBuffer(), (*FileOrErr)->getBuffer());
  ArrayRef<uint8_t> File(
      reinterpret_cast<const uint8_t *>((*FileOrErr)->getBufferStart()),
      (*FileOrErr)->getBufferSize());
  MSFZHeader Header = {};
  std::memcpy(&Header, File.data(), sizeof(Header));
  ASSERT_EQ(Header.NumChunks, 0u);
  ArrayRef<uint8_t> Directory = File.slice(
      Header.StreamDirectoryOffset, Header.StreamDirectorySizeUncompressed);
  support::ulittle64_t StreamLocation;
  std::memcpy(&StreamLocation, Directory.data() + 8, sizeof(StreamLocation));
  EXPECT_EQ(StreamLocation, sizeof(MSFZHeader));
  EXPECT_LE(uint64_t(StreamLocation) + BlockSize, 4096u);

  llvm::copy(Patch, Data.begin());
  EXPECT_EQ(File.slice(uint64_t(StreamLocation), BlockSize), ArrayRef(Data));
}

TEST(MSFZTest, RejectsIncompleteAndNonForwardStreamWrites) {
  constexpr uint32_t BlockSize = 512;
  SuperBlock SB = {};
  SB.BlockSize = BlockSize;
  SB.NumBlocks = 2;

  support::ulittle32_t Sizes[2];
  Sizes[0] = 0;
  Sizes[1] = 8;
  support::ulittle32_t StreamBlock;
  StreamBlock = 1;

  MSFLayout Layout;
  Layout.SB = &SB;
  Layout.StreamSizes = Sizes;
  Layout.StreamMap = {{}, {&StreamBlock, 1}};

  unittest::TempDir TestDirectory("msfz-errors-test", /*Unique=*/true);
  SmallVector<uint8_t, 8> Data(8);

  SmallString<128> InvalidLevelPath = TestDirectory.path("invalid-level.pdz");
  EXPECT_THAT_EXPECTED(
      MSFZWriter::create(InvalidLevelPath, Layout,
                         compression::zstd::getMaxCompressionLevel() + 1),
      Failed());

  SmallString<128> IncompletePath = TestDirectory.path("incomplete.pdz");
  auto IncompleteOrErr = MSFZWriter::create(
      IncompletePath, Layout, compression::zstd::BestSpeedCompression);
  ASSERT_THAT_EXPECTED(IncompleteOrErr, Succeeded());
  std::unique_ptr<MSFZWriter> Incomplete = std::move(*IncompleteOrErr);
  ASSERT_THAT_ERROR(
      Incomplete->writeBytes(BlockSize, ArrayRef(Data).take_front(4)),
      Succeeded());
  EXPECT_THAT_ERROR(Incomplete->finalize(), Failed());

  SmallString<128> NonForwardPath = TestDirectory.path("non-forward.pdz");
  auto NonForwardOrErr = MSFZWriter::create(
      NonForwardPath, Layout, compression::zstd::BestSpeedCompression);
  ASSERT_THAT_EXPECTED(NonForwardOrErr, Succeeded());
  std::unique_ptr<MSFZWriter> NonForward = std::move(*NonForwardOrErr);
  EXPECT_THAT_ERROR(
      NonForward->writeBytes(BlockSize + 1, ArrayRef(Data).drop_front(1)),
      Failed());

  SmallString<128> FinalizedPath = TestDirectory.path("finalized.pdz");
  auto FinalizedOrErr = MSFZWriter::create(
      FinalizedPath, Layout, compression::zstd::BestSpeedCompression);
  ASSERT_THAT_EXPECTED(FinalizedOrErr, Succeeded());
  std::unique_ptr<MSFZWriter> Finalized = std::move(*FinalizedOrErr);
  ASSERT_THAT_ERROR(Finalized->setInitialUncompressedStream(
                        /*StreamIndex=*/1, /*PatchablePrefixSize=*/4),
                    Succeeded());
  ASSERT_THAT_ERROR(Finalized->writeBytes(BlockSize, Data), Succeeded());
  ASSERT_THAT_ERROR(Finalized->finalize(), Succeeded());
  EXPECT_THAT_ERROR(Finalized->patchStream(/*StreamIndex=*/1, /*Offset=*/0,
                                           ArrayRef(Data).take_front(4)),
                    FailedWithMessage("MSFZ output: patch after finalization"));
}
