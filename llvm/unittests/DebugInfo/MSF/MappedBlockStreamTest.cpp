//===- llvm/unittest/DebugInfo/MSF/MappedBlockStreamTest.cpp --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/MSF/MappedBlockStream.h"
#include "llvm/Support/BinaryByteStream.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/BinaryStreamRef.h"
#include "llvm/Support/BinaryStreamWriter.h"
#include "llvm/Testing/Support/Error.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"


using namespace llvm;
using namespace llvm::msf;
using namespace llvm::support;

namespace {

static const uint32_t BlocksAry[] = {0, 1, 2, 5, 4, 3, 6, 7, 8, 9};
static uint8_t DataAry[] = {'A', 'B', 'C', 'F', 'E', 'D', 'G', 'H', 'I', 'J'};

class DiscontiguousStream : public WritableBinaryStream {
public:
  DiscontiguousStream(ArrayRef<uint32_t> Blocks, MutableArrayRef<uint8_t> Data)
      : Blocks(Blocks.begin(), Blocks.end()), Data(Data.begin(), Data.end()) {}

  uint32_t block_size() const { return 1; }
  uint32_t block_count() const { return Blocks.size(); }

  endianness getEndian() const override { return little; }

  Error readBytes(uint64_t Offset, uint64_t Size,
                  ArrayRef<uint8_t> &Buffer) override {
    if (auto EC = checkOffsetForRead(Offset, Size))
      return EC;
    Buffer = Data.slice(Offset, Size);
    return Error::success();
  }

  Error readLongestContiguousChunk(uint64_t Offset,
                                   ArrayRef<uint8_t> &Buffer) override {
    if (auto EC = checkOffsetForRead(Offset, 1))
      return EC;
    Buffer = Data.drop_front(Offset);
    return Error::success();
  }

  uint64_t getLength() override { return Data.size(); }

  Error writeBytes(uint64_t Offset, ArrayRef<uint8_t> SrcData) override {
    if (auto EC = checkOffsetForWrite(Offset, SrcData.size()))
      return EC;
    ::memcpy(&Data[Offset], SrcData.data(), SrcData.size());
    return Error::success();
  }
  Error commit() override { return Error::success(); }

  MSFStreamLayout layout() const {
    return MSFStreamLayout{static_cast<uint32_t>(Data.size()), Blocks};
  }

  BumpPtrAllocator Allocator;

private:
  std::vector<support::ulittle32_t> Blocks;
  MutableArrayRef<uint8_t> Data;
};

TEST(MappedBlockStreamTest, NumBlocks) {
  DiscontiguousStream F(BlocksAry, DataAry);
  auto S = MappedBlockStream::createStream(F.block_size(), F.layout(), F,
                                           F.Allocator);
  EXPECT_EQ(F.block_size(), S->getBlockSize());
  EXPECT_EQ(F.layout().Blocks.size(), S->getNumBlocks());
}

// Tests that a read which is entirely contained within a single block works
// and does not allocate.
TEST(MappedBlockStreamTest, ReadBeyondEndOfStreamRef) {
  DiscontiguousStream F(BlocksAry, DataAry);
  auto S = MappedBlockStream::createStream(F.block_size(), F.layout(), F,
                                           F.Allocator);

  BinaryStreamReader R(*S);
  BinaryStreamRef SR;
  EXPECT_THAT_ERROR(R.readStreamRef(SR, 0U), Succeeded());
  ArrayRef<uint8_t> Buffer;
  EXPECT_THAT_ERROR(SR.readBytes(0U, 1U, Buffer), Failed());
  EXPECT_THAT_ERROR(R.readStreamRef(SR, 1U), Succeeded());
  EXPECT_THAT_ERROR(SR.readBytes(1U, 1U, Buffer), Failed());
}

// Tests that a read which outputs into a full destination buffer works and
// does not fail due to the length of the output buffer.
TEST(MappedBlockStreamTest, ReadOntoNonEmptyBuffer) {
  DiscontiguousStream F(BlocksAry, DataAry);
  auto S = MappedBlockStream::createStream(F.block_size(), F.layout(), F,
                                           F.Allocator);

  BinaryStreamReader R(*S);
  StringRef Str = "ZYXWVUTSRQPONMLKJIHGFEDCBA";
  EXPECT_THAT_ERROR(R.readFixedString(Str, 1), Succeeded());
  EXPECT_EQ(Str, StringRef("A"));
  EXPECT_EQ(0U, F.Allocator.getBytesAllocated());
}

// Tests that a read which crosses a block boundary, but where the subsequent
// blocks are still contiguous in memory to the previous block works and does
// not allocate memory.
TEST(MappedBlockStreamTest, ZeroCopyReadContiguousBreak) {
  DiscontiguousStream F(BlocksAry, DataAry);
  auto S = MappedBlockStream::createStream(F.block_size(), F.layout(), F,
                                           F.Allocator);
  BinaryStreamReader R(*S);
  StringRef Str;
  EXPECT_THAT_ERROR(R.readFixedString(Str, 2), Succeeded());
  EXPECT_EQ(Str, StringRef("AB"));
  EXPECT_EQ(0U, F.Allocator.getBytesAllocated());

  R.setOffset(6);
  EXPECT_THAT_ERROR(R.readFixedString(Str, 4), Succeeded());
  EXPECT_EQ(Str, StringRef("GHIJ"));
  EXPECT_EQ(0U, F.Allocator.getBytesAllocated());
}

// Tests that a read which crosses a block boundary and cannot be referenced
// contiguously works and allocates only the precise amount of bytes
// requested.
TEST(MappedBlockStreamTest, CopyReadNonContiguousBreak) {
  DiscontiguousStream F(BlocksAry, DataAry);
  auto S = MappedBlockStream::createStream(F.block_size(), F.layout(), F,
                                           F.Allocator);
  BinaryStreamReader R(*S);
  StringRef Str;
  EXPECT_THAT_ERROR(R.readFixedString(Str, 10), Succeeded());
  EXPECT_EQ(Str, StringRef("ABCDEFGHIJ"));
  EXPECT_EQ(10U, F.Allocator.getBytesAllocated());
}

// Test that an out of bounds read which doesn't cross a block boundary
// fails and allocates no memory.
TEST(MappedBlockStreamTest, InvalidReadSizeNoBreak) {
  DiscontiguousStream F(BlocksAry, DataAry);
  auto S = MappedBlockStream::createStream(F.block_size(), F.layout(), F,
                                           F.Allocator);
  BinaryStreamReader R(*S);
  StringRef Str;

  R.setOffset(10);
  EXPECT_THAT_ERROR(R.readFixedString(Str, 1), Failed());
  EXPECT_EQ(0U, F.Allocator.getBytesAllocated());
}

// Test that an out of bounds read which crosses a contiguous block boundary
// fails and allocates no memory.
TEST(MappedBlockStreamTest, InvalidReadSizeContiguousBreak) {
  DiscontiguousStream F(BlocksAry, DataAry);
  auto S = MappedBlockStream::createStream(F.block_size(), F.layout(), F,
                                           F.Allocator);
  BinaryStreamReader R(*S);
  StringRef Str;

  R.setOffset(6);
  EXPECT_THAT_ERROR(R.readFixedString(Str, 5), Failed());
  EXPECT_EQ(0U, F.Allocator.getBytesAllocated());
}

// Test that an out of bounds read which crosses a discontiguous block
// boundary fails and allocates no memory.
TEST(MappedBlockStreamTest, InvalidReadSizeNonContiguousBreak) {
  DiscontiguousStream F(BlocksAry, DataAry);
  auto S = MappedBlockStream::createStream(F.block_size(), F.layout(), F,
                                           F.Allocator);
  BinaryStreamReader R(*S);
  StringRef Str;

  EXPECT_THAT_ERROR(R.readFixedString(Str, 11), Failed());
  EXPECT_EQ(0U, F.Allocator.getBytesAllocated());
}

// Tests that a read which is entirely contained within a single block but
// beyond the end of a StreamRef fails.
TEST(MappedBlockStreamTest, ZeroCopyReadNoBreak) {
  DiscontiguousStream F(BlocksAry, DataAry);
  auto S = MappedBlockStream::createStream(F.block_size(), F.layout(), F,
                                           F.Allocator);
  BinaryStreamReader R(*S);
  StringRef Str;
  EXPECT_THAT_ERROR(R.readFixedString(Str, 1), Succeeded());
  EXPECT_EQ(Str, StringRef("A"));
  EXPECT_EQ(0U, F.Allocator.getBytesAllocated());
}

// Tests that a read which is not aligned on the same boundary as a previous
// cached request, but which is known to overlap that request, shares the
// previous allocation.
TEST(MappedBlockStreamTest, UnalignedOverlappingRead) {
  DiscontiguousStream F(BlocksAry, DataAry);
  auto S = MappedBlockStream::createStream(F.block_size(), F.layout(), F,
                                           F.Allocator);
  BinaryStreamReader R(*S);
  StringRef Str1;
  StringRef Str2;
  EXPECT_THAT_ERROR(R.readFixedString(Str1, 7), Succeeded());
  EXPECT_EQ(Str1, StringRef("ABCDEFG"));
  EXPECT_EQ(7U, F.Allocator.getBytesAllocated());

  R.setOffset(2);
  EXPECT_THAT_ERROR(R.readFixedString(Str2, 3), Succeeded());
  EXPECT_EQ(Str2, StringRef("CDE"));
  EXPECT_EQ(Str1.data() + 2, Str2.data());
  EXPECT_EQ(7U, F.Allocator.getBytesAllocated());
}

// Tests that a read which is not aligned on the same boundary as a previous
// cached request, but which only partially overlaps a previous cached request,
// still works correctly and allocates again from the shared pool.
TEST(MappedBlockStreamTest, UnalignedOverlappingReadFail) {
  DiscontiguousStream F(BlocksAry, DataAry);
  auto S = MappedBlockStream::createStream(F.block_size(), F.layout(), F,
                                           F.Allocator);
  BinaryStreamReader R(*S);
  StringRef Str1;
  StringRef Str2;
  EXPECT_THAT_ERROR(R.readFixedString(Str1, 6), Succeeded());
  EXPECT_EQ(Str1, StringRef("ABCDEF"));
  EXPECT_EQ(6U, F.Allocator.getBytesAllocated());

  R.setOffset(4);
  EXPECT_THAT_ERROR(R.readFixedString(Str2, 4), Succeeded());
  EXPECT_EQ(Str2, StringRef("EFGH"));
  EXPECT_EQ(10U, F.Allocator.getBytesAllocated());
}

TEST(MappedBlockStreamTest, WriteBeyondEndOfStream) {
  static uint8_t Data[] = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'};
  static uint8_t LargeBuffer[] = {'0', '1', '2', '3', '4', '5',
                                  '6', '7', '8', '9', 'A'};
  static uint8_t SmallBuffer[] = {'0', '1', '2'};
  static_assert(sizeof(LargeBuffer) > sizeof(Data),
                "LargeBuffer is not big enough");

  DiscontiguousStream F(BlocksAry, Data);
  auto S = WritableMappedBlockStream::createStream(F.block_size(), F.layout(),
                                                   F, F.Allocator);
  EXPECT_THAT_ERROR(S->writeBytes(0, ArrayRef<uint8_t>(LargeBuffer)), Failed());
  EXPECT_THAT_ERROR(S->writeBytes(0, ArrayRef<uint8_t>(SmallBuffer)),
                    Succeeded());
  EXPECT_THAT_ERROR(S->writeBytes(7, ArrayRef<uint8_t>(SmallBuffer)),
                    Succeeded());
  EXPECT_THAT_ERROR(S->writeBytes(8, ArrayRef<uint8_t>(SmallBuffer)), Failed());
}

TEST(MappedBlockStreamTest, TestWriteBytesNoBreakBoundary) {
  static uint8_t Data[] = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'};
  DiscontiguousStream F(BlocksAry, Data);
  auto S = WritableMappedBlockStream::createStream(F.block_size(), F.layout(),
                                                   F, F.Allocator);
  ArrayRef<uint8_t> Buffer;

  EXPECT_THAT_ERROR(S->readBytes(0, 1, Buffer), Succeeded());
  EXPECT_EQ(Buffer, ArrayRef<uint8_t>('A'));
  EXPECT_THAT_ERROR(S->readBytes(9, 1, Buffer), Succeeded());
  EXPECT_EQ(Buffer, ArrayRef<uint8_t>('J'));

  EXPECT_THAT_ERROR(S->writeBytes(0, ArrayRef<uint8_t>('J')), Succeeded());
  EXPECT_THAT_ERROR(S->writeBytes(9, ArrayRef<uint8_t>('A')), Succeeded());

  EXPECT_THAT_ERROR(S->readBytes(0, 1, Buffer), Succeeded());
  EXPECT_EQ(Buffer, ArrayRef<uint8_t>('J'));
  EXPECT_THAT_ERROR(S->readBytes(9, 1, Buffer), Succeeded());
  EXPECT_EQ(Buffer, ArrayRef<uint8_t>('A'));

  EXPECT_THAT_ERROR(S->writeBytes(0, ArrayRef<uint8_t>('A')), Succeeded());
  EXPECT_THAT_ERROR(S->writeBytes(9, ArrayRef<uint8_t>('J')), Succeeded());

  EXPECT_THAT_ERROR(S->readBytes(0, 1, Buffer), Succeeded());
  EXPECT_EQ(Buffer, ArrayRef<uint8_t>('A'));
  EXPECT_THAT_ERROR(S->readBytes(9, 1, Buffer), Succeeded());
  EXPECT_EQ(Buffer, ArrayRef<uint8_t>('J'));
}

TEST(MappedBlockStreamTest, TestWriteBytesBreakBoundary) {
  static uint8_t Data[] = {'0', '0', '0', '0', '0', '0', '0', '0', '0', '0'};
  static uint8_t TestData[] = {'T', 'E', 'S', 'T', 'I', 'N', 'G', '.'};
  static uint8_t Expected[] = {'T', 'E', 'S', 'N', 'I',
                               'T', 'G', '.', '0', '0'};

  DiscontiguousStream F(BlocksAry, Data);
  auto S = WritableMappedBlockStream::createStream(F.block_size(), F.layout(),
                                                   F, F.Allocator);
  ArrayRef<uint8_t> Buffer;

  EXPECT_THAT_ERROR(S->writeBytes(0, TestData), Succeeded());
  // First just compare the memory, then compare the result of reading the
  // string out.
  EXPECT_EQ(ArrayRef<uint8_t>(Data), ArrayRef<uint8_t>(Expected));

  EXPECT_THAT_ERROR(S->readBytes(0, 8, Buffer), Succeeded());
  EXPECT_EQ(Buffer, ArrayRef<uint8_t>(TestData));
}

TEST(MappedBlockStreamTest, TestWriteThenRead) {
  std::vector<uint8_t> DataBytes(10);
  MutableArrayRef<uint8_t> Data(DataBytes);
  const uint32_t Blocks[] = {2, 1, 0, 6, 3, 4, 5, 7, 9, 8};

  DiscontiguousStream F(Blocks, Data);
  auto S = WritableMappedBlockStream::createStream(F.block_size(), F.layout(),
                                                   F, F.Allocator);

  enum class MyEnum : uint32_t { Val1 = 2908234, Val2 = 120891234 };
  using support::ulittle32_t;

  uint16_t u16[] = {31468, 0};
  uint32_t u32[] = {890723408, 0};
  MyEnum Enum[] = {MyEnum::Val1, MyEnum::Val2};
  StringRef ZStr[] = {"Zero Str", ""};
  StringRef FStr[] = {"Fixed Str", ""};
  uint8_t byteArray0[] = {'1', '2'};
  uint8_t byteArray1[] = {'0', '0'};
  ArrayRef<uint8_t> byteArrayRef0(byteArray0);
  ArrayRef<uint8_t> byteArrayRef1(byteArray1);
  ArrayRef<uint8_t> byteArray[] = {byteArrayRef0, byteArrayRef1};
  uint32_t intArr0[] = {890723408, 29082234};
  uint32_t intArr1[] = {890723408, 29082234};
  ArrayRef<uint32_t> intArray[] = {intArr0, intArr1};

  BinaryStreamReader Reader(*S);
  BinaryStreamWriter Writer(*S);
  EXPECT_THAT_ERROR(Writer.writeInteger(u16[0]), Succeeded());
  EXPECT_THAT_ERROR(Reader.readInteger(u16[1]), Succeeded());
  EXPECT_EQ(u16[0], u16[1]);
  EXPECT_EQ(std::vector<uint8_t>({0, 0x7A, 0xEC, 0, 0, 0, 0, 0, 0, 0}),
            DataBytes);

  Reader.setOffset(0);
  Writer.setOffset(0);
  ::memset(DataBytes.data(), 0, 10);
  EXPECT_THAT_ERROR(Writer.writeInteger(u32[0]), Succeeded());
  EXPECT_THAT_ERROR(Reader.readInteger(u32[1]), Succeeded());
  EXPECT_EQ(u32[0], u32[1]);
  EXPECT_EQ(std::vector<uint8_t>({0x17, 0x5C, 0x50, 0, 0, 0, 0x35, 0, 0, 0}),
            DataBytes);

  Reader.setOffset(0);
  Writer.setOffset(0);
  ::memset(DataBytes.data(), 0, 10);
  EXPECT_THAT_ERROR(Writer.writeEnum(Enum[0]), Succeeded());
  EXPECT_THAT_ERROR(Reader.readEnum(Enum[1]), Succeeded());
  EXPECT_EQ(Enum[0], Enum[1]);
  EXPECT_EQ(std::vector<uint8_t>({0x2C, 0x60, 0x4A, 0, 0, 0, 0, 0, 0, 0}),
            DataBytes);

  Reader.setOffset(0);
  Writer.setOffset(0);
  ::memset(DataBytes.data(), 0, 10);
  EXPECT_THAT_ERROR(Writer.writeCString(ZStr[0]), Succeeded());
  EXPECT_THAT_ERROR(Reader.readCString(ZStr[1]), Succeeded());
  EXPECT_EQ(ZStr[0], ZStr[1]);
  EXPECT_EQ(
      std::vector<uint8_t>({'r', 'e', 'Z', ' ', 'S', 't', 'o', 'r', 0, 0}),
      DataBytes);

  Reader.setOffset(0);
  Writer.setOffset(0);
  ::memset(DataBytes.data(), 0, 10);
  EXPECT_THAT_ERROR(Writer.writeFixedString(FStr[0]), Succeeded());
  EXPECT_THAT_ERROR(Reader.readFixedString(FStr[1], FStr[0].size()),
                    Succeeded());
  EXPECT_EQ(FStr[0], FStr[1]);
  EXPECT_EQ(
      std::vector<uint8_t>({'x', 'i', 'F', 'd', ' ', 'S', 'e', 't', 0, 'r'}),
      DataBytes);

  Reader.setOffset(0);
  Writer.setOffset(0);
  ::memset(DataBytes.data(), 0, 10);
  EXPECT_THAT_ERROR(Writer.writeArray(byteArray[0]), Succeeded());
  EXPECT_THAT_ERROR(Reader.readArray(byteArray[1], byteArray[0].size()),
                    Succeeded());
  EXPECT_EQ(byteArray[0], byteArray[1]);
  EXPECT_EQ(std::vector<uint8_t>({0, 0x32, 0x31, 0, 0, 0, 0, 0, 0, 0}),
            DataBytes);

  Reader.setOffset(0);
  Writer.setOffset(0);
  ::memset(DataBytes.data(), 0, 10);
  EXPECT_THAT_ERROR(Writer.writeArray(intArray[0]), Succeeded());
  EXPECT_THAT_ERROR(Reader.readArray(intArray[1], intArray[0].size()),
                    Succeeded());
  EXPECT_EQ(intArray[0], intArray[1]);
}

TEST(MappedBlockStreamTest, TestWriteContiguousStreamRef) {
  std::vector<uint8_t> DestDataBytes(10);
  MutableArrayRef<uint8_t> DestData(DestDataBytes);
  const uint32_t DestBlocks[] = {2, 1, 0, 6, 3, 4, 5, 7, 9, 8};

  std::vector<uint8_t> SrcDataBytes(10);
  MutableArrayRef<uint8_t> SrcData(SrcDataBytes);

  DiscontiguousStream F(DestBlocks, DestData);
  auto DestStream = WritableMappedBlockStream::createStream(
      F.block_size(), F.layout(), F, F.Allocator);

  // First write "Test Str" into the source stream.
  MutableBinaryByteStream SourceStream(SrcData, little);
  BinaryStreamWriter SourceWriter(SourceStream);
  EXPECT_THAT_ERROR(SourceWriter.writeCString("Test Str"), Succeeded());
  EXPECT_EQ(SrcDataBytes, std::vector<uint8_t>(
                              {'T', 'e', 's', 't', ' ', 'S', 't', 'r', 0, 0}));

  // Then write the source stream into the dest stream.
  BinaryStreamWriter DestWriter(*DestStream);
  EXPECT_THAT_ERROR(DestWriter.writeStreamRef(SourceStream), Succeeded());
  EXPECT_EQ(DestDataBytes, std::vector<uint8_t>(
                               {'s', 'e', 'T', ' ', 'S', 't', 't', 'r', 0, 0}));

  // Then read the string back out of the dest stream.
  StringRef Result;
  BinaryStreamReader DestReader(*DestStream);
  EXPECT_THAT_ERROR(DestReader.readCString(Result), Succeeded());
  EXPECT_EQ(Result, "Test Str");
}

TEST(MappedBlockStreamTest, TestWriteDiscontiguousStreamRef) {
  std::vector<uint8_t> DestDataBytes(10);
  MutableArrayRef<uint8_t> DestData(DestDataBytes);
  const uint32_t DestBlocks[] = {2, 1, 0, 6, 3, 4, 5, 7, 9, 8};

  std::vector<uint8_t> SrcDataBytes(10);
  MutableArrayRef<uint8_t> SrcData(SrcDataBytes);
  const uint32_t SrcBlocks[] = {1, 0, 6, 3, 4, 5, 2, 7, 8, 9};

  DiscontiguousStream DestF(DestBlocks, DestData);
  DiscontiguousStream SrcF(SrcBlocks, SrcData);

  auto Dest = WritableMappedBlockStream::createStream(
      DestF.block_size(), DestF.layout(), DestF, DestF.Allocator);
  auto Src = WritableMappedBlockStream::createStream(
      SrcF.block_size(), SrcF.layout(), SrcF, SrcF.Allocator);

  // First write "Test Str" into the source stream.
  BinaryStreamWriter SourceWriter(*Src);
  EXPECT_THAT_ERROR(SourceWriter.writeCString("Test Str"), Succeeded());
  EXPECT_EQ(SrcDataBytes, std::vector<uint8_t>(
                              {'e', 'T', 't', 't', ' ', 'S', 's', 'r', 0, 0}));

  // Then write the source stream into the dest stream.
  BinaryStreamWriter DestWriter(*Dest);
  EXPECT_THAT_ERROR(DestWriter.writeStreamRef(*Src), Succeeded());
  EXPECT_EQ(DestDataBytes, std::vector<uint8_t>(
                               {'s', 'e', 'T', ' ', 'S', 't', 't', 'r', 0, 0}));

  // Then read the string back out of the dest stream.
  StringRef Result;
  BinaryStreamReader DestReader(*Dest);
  EXPECT_THAT_ERROR(DestReader.readCString(Result), Succeeded());
  EXPECT_EQ(Result, "Test Str");
}

TEST(MappedBlockStreamTest, DataLivesAfterStreamDestruction) {
  std::vector<uint8_t> DataBytes(10);
  MutableArrayRef<uint8_t> Data(DataBytes);
  const uint32_t Blocks[] = {2, 1, 0, 6, 3, 4, 5, 7, 9, 8};

  StringRef Str[] = {"Zero Str", ""};

  DiscontiguousStream F(Blocks, Data);
  {
    auto S = WritableMappedBlockStream::createStream(F.block_size(), F.layout(),
                                                     F, F.Allocator);

    BinaryStreamReader Reader(*S);
    BinaryStreamWriter Writer(*S);
    ::memset(DataBytes.data(), 0, 10);
    EXPECT_THAT_ERROR(Writer.writeCString(Str[0]), Succeeded());
    EXPECT_THAT_ERROR(Reader.readCString(Str[1]), Succeeded());
    EXPECT_EQ(Str[0], Str[1]);
  }

  EXPECT_EQ(Str[0], Str[1]);
}
} // namespace

MATCHER_P3(BlockIsFilledWith, Layout, BlockIndex, Byte, "succeeded") {
  uint64_t Offset = msf::blockToOffset(BlockIndex, Layout.SB->BlockSize);
  ArrayRef<uint8_t> BufferRef = makeArrayRef(arg);
  BufferRef = BufferRef.slice(Offset, Layout.SB->BlockSize);
  return llvm::all_of(BufferRef, [this](uint8_t B) { return B == Byte; });
}

namespace {
TEST(MappedBlockStreamTest, CreateFpmStream) {
  BumpPtrAllocator Allocator;
  SuperBlock SB;
  MSFLayout L;
  L.SB = &SB;

  SB.FreeBlockMapBlock = 1;
  SB.BlockSize = 4096;

  constexpr uint32_t NumFileBlocks = 4096 * 4;

  std::vector<uint8_t> MsfBuffer(NumFileBlocks * SB.BlockSize);
  MutableBinaryByteStream MsfStream(MsfBuffer, llvm::support::little);

  SB.NumBlocks = NumFileBlocks;
  auto FpmStream =
      WritableMappedBlockStream::createFpmStream(L, MsfStream, Allocator);
  // 4096 * 4 / 8 = 2048 bytes of FPM data is needed to describe 4096 * 4
  // blocks.  This translates to 1 FPM block.
  EXPECT_EQ(2048u, FpmStream->getLength());
  EXPECT_EQ(1u, FpmStream->getStreamLayout().Blocks.size());
  EXPECT_EQ(1u, FpmStream->getStreamLayout().Blocks[0]);
  // All blocks from FPM1 should be 1 initialized, and all blocks from FPM2
  // should be 0 initialized (since we requested the main FPM, not the alt FPM)
  for (int I = 0; I < 4; ++I) {
    EXPECT_THAT(MsfBuffer, BlockIsFilledWith(L, 1 + I * SB.BlockSize, 0xFF));
    EXPECT_THAT(MsfBuffer, BlockIsFilledWith(L, 2 + I * SB.BlockSize, 0));
  }

  ::memset(MsfBuffer.data(), 0, MsfBuffer.size());
  FpmStream =
      WritableMappedBlockStream::createFpmStream(L, MsfStream, Allocator, true);
  // 4096 * 4 / 8 = 2048 bytes of FPM data is needed to describe 4096 * 4
  // blocks.  This translates to 1 FPM block.
  EXPECT_EQ(2048u, FpmStream->getLength());
  EXPECT_EQ(1u, FpmStream->getStreamLayout().Blocks.size());
  EXPECT_EQ(2u, FpmStream->getStreamLayout().Blocks[0]);
  // All blocks from FPM2 should be 1 initialized, and all blocks from FPM1
  // should be 0 initialized (since we requested the alt FPM, not the main FPM)
  for (int I = 0; I < 4; ++I) {
    EXPECT_THAT(MsfBuffer, BlockIsFilledWith(L, 1 + I * SB.BlockSize, 0));
    EXPECT_THAT(MsfBuffer, BlockIsFilledWith(L, 2 + I * SB.BlockSize, 0xFF));
  }
}

} // end anonymous namespace
