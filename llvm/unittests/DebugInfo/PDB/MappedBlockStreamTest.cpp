//===- llvm/unittest/DebugInfo/PDB/MappedBlockStreamTest.cpp --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ErrorChecking.h"

#include "llvm/DebugInfo/MSF/BinaryByteStream.h"
#include "llvm/DebugInfo/MSF/BinaryStreamReader.h"
#include "llvm/DebugInfo/MSF/BinaryStreamRef.h"
#include "llvm/DebugInfo/MSF/BinaryStreamWriter.h"
#include "llvm/DebugInfo/MSF/IMSFFile.h"
#include "llvm/DebugInfo/MSF/MSFError.h"
#include "llvm/DebugInfo/MSF/MSFStreamLayout.h"
#include "llvm/DebugInfo/MSF/MappedBlockStream.h"
#include "gtest/gtest.h"

#include <unordered_map>

using namespace llvm;
using namespace llvm::msf;

namespace {

static const uint32_t BlocksAry[] = {0, 1, 2, 5, 4, 3, 6, 7, 8, 9};
static uint8_t DataAry[] = {'A', 'B', 'C', 'F', 'E', 'D', 'G', 'H', 'I', 'J'};

class DiscontiguousStream : public WritableBinaryStream {
public:
  DiscontiguousStream(ArrayRef<uint32_t> Blocks, MutableArrayRef<uint8_t> Data)
      : Blocks(Blocks.begin(), Blocks.end()), Data(Data.begin(), Data.end()) {}

  uint32_t block_size() const { return 1; }
  uint32_t block_count() const { return Blocks.size(); }

  Error readBytes(uint32_t Offset, uint32_t Size,
                  ArrayRef<uint8_t> &Buffer) override {
    if (Offset + Size > Data.size())
      return make_error<MSFError>(msf_error_code::insufficient_buffer);
    Buffer = Data.slice(Offset, Size);
    return Error::success();
  }

  Error readLongestContiguousChunk(uint32_t Offset,
                                   ArrayRef<uint8_t> &Buffer) override {
    if (Offset >= Data.size())
      return make_error<MSFError>(msf_error_code::insufficient_buffer);
    Buffer = Data.drop_front(Offset);
    return Error::success();
  }

  uint32_t getLength() override { return Data.size(); }

  Error writeBytes(uint32_t Offset, ArrayRef<uint8_t> SrcData) override {
    if (Offset + SrcData.size() > Data.size())
      return make_error<MSFError>(msf_error_code::insufficient_buffer);
    ::memcpy(&Data[Offset], SrcData.data(), SrcData.size());
    return Error::success();
  }
  Error commit() override { return Error::success(); }

  MSFStreamLayout layout() const {
    return MSFStreamLayout{static_cast<uint32_t>(Data.size()), Blocks};
  }

private:
  std::vector<support::ulittle32_t> Blocks;
  MutableArrayRef<uint8_t> Data;
};

// Tests that a read which is entirely contained within a single block works
// and does not allocate.
TEST(MappedBlockStreamTest, ReadBeyondEndOfStreamRef) {
  DiscontiguousStream F(BlocksAry, DataAry);
  auto S = MappedBlockStream::createStream(F.block_size(), F.block_count(),
                                           F.layout(), F);

  BinaryStreamReader R(*S);
  BinaryStreamRef SR;
  EXPECT_NO_ERROR(R.readStreamRef(SR, 0U));
  ArrayRef<uint8_t> Buffer;
  EXPECT_ERROR(SR.readBytes(0U, 1U, Buffer));
  EXPECT_NO_ERROR(R.readStreamRef(SR, 1U));
  EXPECT_ERROR(SR.readBytes(1U, 1U, Buffer));
}

// Tests that a read which outputs into a full destination buffer works and
// does not fail due to the length of the output buffer.
TEST(MappedBlockStreamTest, ReadOntoNonEmptyBuffer) {
  DiscontiguousStream F(BlocksAry, DataAry);
  auto S = MappedBlockStream::createStream(F.block_size(), F.block_count(),
                                           F.layout(), F);

  BinaryStreamReader R(*S);
  StringRef Str = "ZYXWVUTSRQPONMLKJIHGFEDCBA";
  EXPECT_NO_ERROR(R.readFixedString(Str, 1));
  EXPECT_EQ(Str, StringRef("A"));
  EXPECT_EQ(0U, S->getNumBytesCopied());
}

// Tests that a read which crosses a block boundary, but where the subsequent
// blocks are still contiguous in memory to the previous block works and does
// not allocate memory.
TEST(MappedBlockStreamTest, ZeroCopyReadContiguousBreak) {
  DiscontiguousStream F(BlocksAry, DataAry);
  auto S = MappedBlockStream::createStream(F.block_size(), F.block_count(),
                                           F.layout(), F);
  BinaryStreamReader R(*S);
  StringRef Str;
  EXPECT_NO_ERROR(R.readFixedString(Str, 2));
  EXPECT_EQ(Str, StringRef("AB"));
  EXPECT_EQ(0U, S->getNumBytesCopied());

  R.setOffset(6);
  EXPECT_NO_ERROR(R.readFixedString(Str, 4));
  EXPECT_EQ(Str, StringRef("GHIJ"));
  EXPECT_EQ(0U, S->getNumBytesCopied());
}

// Tests that a read which crosses a block boundary and cannot be referenced
// contiguously works and allocates only the precise amount of bytes
// requested.
TEST(MappedBlockStreamTest, CopyReadNonContiguousBreak) {
  DiscontiguousStream F(BlocksAry, DataAry);
  auto S = MappedBlockStream::createStream(F.block_size(), F.block_count(),
                                           F.layout(), F);
  BinaryStreamReader R(*S);
  StringRef Str;
  EXPECT_NO_ERROR(R.readFixedString(Str, 10));
  EXPECT_EQ(Str, StringRef("ABCDEFGHIJ"));
  EXPECT_EQ(10U, S->getNumBytesCopied());
}

// Test that an out of bounds read which doesn't cross a block boundary
// fails and allocates no memory.
TEST(MappedBlockStreamTest, InvalidReadSizeNoBreak) {
  DiscontiguousStream F(BlocksAry, DataAry);
  auto S = MappedBlockStream::createStream(F.block_size(), F.block_count(),
                                           F.layout(), F);
  BinaryStreamReader R(*S);
  StringRef Str;

  R.setOffset(10);
  EXPECT_ERROR(R.readFixedString(Str, 1));
  EXPECT_EQ(0U, S->getNumBytesCopied());
}

// Test that an out of bounds read which crosses a contiguous block boundary
// fails and allocates no memory.
TEST(MappedBlockStreamTest, InvalidReadSizeContiguousBreak) {
  DiscontiguousStream F(BlocksAry, DataAry);
  auto S = MappedBlockStream::createStream(F.block_size(), F.block_count(),
                                           F.layout(), F);
  BinaryStreamReader R(*S);
  StringRef Str;

  R.setOffset(6);
  EXPECT_ERROR(R.readFixedString(Str, 5));
  EXPECT_EQ(0U, S->getNumBytesCopied());
}

// Test that an out of bounds read which crosses a discontiguous block
// boundary fails and allocates no memory.
TEST(MappedBlockStreamTest, InvalidReadSizeNonContiguousBreak) {
  DiscontiguousStream F(BlocksAry, DataAry);
  auto S = MappedBlockStream::createStream(F.block_size(), F.block_count(),
                                           F.layout(), F);
  BinaryStreamReader R(*S);
  StringRef Str;

  EXPECT_ERROR(R.readFixedString(Str, 11));
  EXPECT_EQ(0U, S->getNumBytesCopied());
}

// Tests that a read which is entirely contained within a single block but
// beyond the end of a StreamRef fails.
TEST(MappedBlockStreamTest, ZeroCopyReadNoBreak) {
  DiscontiguousStream F(BlocksAry, DataAry);
  auto S = MappedBlockStream::createStream(F.block_size(), F.block_count(),
                                           F.layout(), F);
  BinaryStreamReader R(*S);
  StringRef Str;
  EXPECT_NO_ERROR(R.readFixedString(Str, 1));
  EXPECT_EQ(Str, StringRef("A"));
  EXPECT_EQ(0U, S->getNumBytesCopied());
}

// Tests that a read which is not aligned on the same boundary as a previous
// cached request, but which is known to overlap that request, shares the
// previous allocation.
TEST(MappedBlockStreamTest, UnalignedOverlappingRead) {
  DiscontiguousStream F(BlocksAry, DataAry);
  auto S = MappedBlockStream::createStream(F.block_size(), F.block_count(),
                                           F.layout(), F);
  BinaryStreamReader R(*S);
  StringRef Str1;
  StringRef Str2;
  EXPECT_NO_ERROR(R.readFixedString(Str1, 7));
  EXPECT_EQ(Str1, StringRef("ABCDEFG"));
  EXPECT_EQ(7U, S->getNumBytesCopied());

  R.setOffset(2);
  EXPECT_NO_ERROR(R.readFixedString(Str2, 3));
  EXPECT_EQ(Str2, StringRef("CDE"));
  EXPECT_EQ(Str1.data() + 2, Str2.data());
  EXPECT_EQ(7U, S->getNumBytesCopied());
}

// Tests that a read which is not aligned on the same boundary as a previous
// cached request, but which only partially overlaps a previous cached request,
// still works correctly and allocates again from the shared pool.
TEST(MappedBlockStreamTest, UnalignedOverlappingReadFail) {
  DiscontiguousStream F(BlocksAry, DataAry);
  auto S = MappedBlockStream::createStream(F.block_size(), F.block_count(),
                                           F.layout(), F);
  BinaryStreamReader R(*S);
  StringRef Str1;
  StringRef Str2;
  EXPECT_NO_ERROR(R.readFixedString(Str1, 6));
  EXPECT_EQ(Str1, StringRef("ABCDEF"));
  EXPECT_EQ(6U, S->getNumBytesCopied());

  R.setOffset(4);
  EXPECT_NO_ERROR(R.readFixedString(Str2, 4));
  EXPECT_EQ(Str2, StringRef("EFGH"));
  EXPECT_EQ(10U, S->getNumBytesCopied());
}

TEST(MappedBlockStreamTest, WriteBeyondEndOfStream) {
  static uint8_t Data[] = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'};
  static uint8_t LargeBuffer[] = {'0', '1', '2', '3', '4', '5',
                                  '6', '7', '8', '9', 'A'};
  static uint8_t SmallBuffer[] = {'0', '1', '2'};
  static_assert(sizeof(LargeBuffer) > sizeof(Data),
                "LargeBuffer is not big enough");

  DiscontiguousStream F(BlocksAry, Data);
  auto S = WritableMappedBlockStream::createStream(
      F.block_size(), F.block_count(), F.layout(), F);
  ArrayRef<uint8_t> Buffer;

  EXPECT_ERROR(S->writeBytes(0, ArrayRef<uint8_t>(LargeBuffer)));
  EXPECT_NO_ERROR(S->writeBytes(0, ArrayRef<uint8_t>(SmallBuffer)));
  EXPECT_NO_ERROR(S->writeBytes(7, ArrayRef<uint8_t>(SmallBuffer)));
  EXPECT_ERROR(S->writeBytes(8, ArrayRef<uint8_t>(SmallBuffer)));
}

TEST(MappedBlockStreamTest, TestWriteBytesNoBreakBoundary) {
  static uint8_t Data[] = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'};
  DiscontiguousStream F(BlocksAry, Data);
  auto S = WritableMappedBlockStream::createStream(
      F.block_size(), F.block_count(), F.layout(), F);
  ArrayRef<uint8_t> Buffer;

  EXPECT_NO_ERROR(S->readBytes(0, 1, Buffer));
  EXPECT_EQ(Buffer, ArrayRef<uint8_t>('A'));
  EXPECT_NO_ERROR(S->readBytes(9, 1, Buffer));
  EXPECT_EQ(Buffer, ArrayRef<uint8_t>('J'));

  EXPECT_NO_ERROR(S->writeBytes(0, ArrayRef<uint8_t>('J')));
  EXPECT_NO_ERROR(S->writeBytes(9, ArrayRef<uint8_t>('A')));

  EXPECT_NO_ERROR(S->readBytes(0, 1, Buffer));
  EXPECT_EQ(Buffer, ArrayRef<uint8_t>('J'));
  EXPECT_NO_ERROR(S->readBytes(9, 1, Buffer));
  EXPECT_EQ(Buffer, ArrayRef<uint8_t>('A'));

  EXPECT_NO_ERROR(S->writeBytes(0, ArrayRef<uint8_t>('A')));
  EXPECT_NO_ERROR(S->writeBytes(9, ArrayRef<uint8_t>('J')));

  EXPECT_NO_ERROR(S->readBytes(0, 1, Buffer));
  EXPECT_EQ(Buffer, ArrayRef<uint8_t>('A'));
  EXPECT_NO_ERROR(S->readBytes(9, 1, Buffer));
  EXPECT_EQ(Buffer, ArrayRef<uint8_t>('J'));
}

TEST(MappedBlockStreamTest, TestWriteBytesBreakBoundary) {
  static uint8_t Data[] = {'0', '0', '0', '0', '0', '0', '0', '0', '0', '0'};
  static uint8_t TestData[] = {'T', 'E', 'S', 'T', 'I', 'N', 'G', '.'};
  static uint8_t Expected[] = {'T', 'E', 'S', 'N', 'I',
                               'T', 'G', '.', '0', '0'};

  DiscontiguousStream F(BlocksAry, Data);
  auto S = WritableMappedBlockStream::createStream(
      F.block_size(), F.block_count(), F.layout(), F);
  ArrayRef<uint8_t> Buffer;

  EXPECT_NO_ERROR(S->writeBytes(0, TestData));
  // First just compare the memory, then compare the result of reading the
  // string out.
  EXPECT_EQ(ArrayRef<uint8_t>(Data), ArrayRef<uint8_t>(Expected));

  EXPECT_NO_ERROR(S->readBytes(0, 8, Buffer));
  EXPECT_EQ(Buffer, ArrayRef<uint8_t>(TestData));
}

TEST(MappedBlockStreamTest, TestWriteThenRead) {
  std::vector<uint8_t> DataBytes(10);
  MutableArrayRef<uint8_t> Data(DataBytes);
  const uint32_t Blocks[] = {2, 1, 0, 6, 3, 4, 5, 7, 9, 8};

  DiscontiguousStream F(Blocks, Data);
  auto S = WritableMappedBlockStream::createStream(
      F.block_size(), F.block_count(), F.layout(), F);

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
  ArrayRef<uint8_t> byteArray[] = { byteArrayRef0, byteArrayRef1 };
  uint32_t intArr0[] = {890723408, 29082234};
  uint32_t intArr1[] = {890723408, 29082234};
  ArrayRef<uint32_t> intArray[] = {intArr0, intArr1};

  BinaryStreamReader Reader(*S);
  BinaryStreamWriter Writer(*S);
  EXPECT_NO_ERROR(Writer.writeInteger(u16[0], llvm::support::little));
  EXPECT_NO_ERROR(Reader.readInteger(u16[1], llvm::support::little));
  EXPECT_EQ(u16[0], u16[1]);
  EXPECT_EQ(std::vector<uint8_t>({0, 0x7A, 0xEC, 0, 0, 0, 0, 0, 0, 0}),
            DataBytes);

  Reader.setOffset(0);
  Writer.setOffset(0);
  ::memset(DataBytes.data(), 0, 10);
  EXPECT_NO_ERROR(Writer.writeInteger(u32[0], llvm::support::little));
  EXPECT_NO_ERROR(Reader.readInteger(u32[1], llvm::support::little));
  EXPECT_EQ(u32[0], u32[1]);
  EXPECT_EQ(std::vector<uint8_t>({0x17, 0x5C, 0x50, 0, 0, 0, 0x35, 0, 0, 0}),
            DataBytes);

  Reader.setOffset(0);
  Writer.setOffset(0);
  ::memset(DataBytes.data(), 0, 10);
  EXPECT_NO_ERROR(Writer.writeEnum(Enum[0], llvm::support::little));
  EXPECT_NO_ERROR(Reader.readEnum(Enum[1], llvm::support::little));
  EXPECT_EQ(Enum[0], Enum[1]);
  EXPECT_EQ(std::vector<uint8_t>({0x2C, 0x60, 0x4A, 0, 0, 0, 0, 0, 0, 0}),
            DataBytes);

  Reader.setOffset(0);
  Writer.setOffset(0);
  ::memset(DataBytes.data(), 0, 10);
  EXPECT_NO_ERROR(Writer.writeCString(ZStr[0]));
  EXPECT_NO_ERROR(Reader.readCString(ZStr[1]));
  EXPECT_EQ(ZStr[0], ZStr[1]);
  EXPECT_EQ(
      std::vector<uint8_t>({'r', 'e', 'Z', ' ', 'S', 't', 'o', 'r', 0, 0}),
      DataBytes);

  Reader.setOffset(0);
  Writer.setOffset(0);
  ::memset(DataBytes.data(), 0, 10);
  EXPECT_NO_ERROR(Writer.writeFixedString(FStr[0]));
  EXPECT_NO_ERROR(Reader.readFixedString(FStr[1], FStr[0].size()));
  EXPECT_EQ(FStr[0], FStr[1]);
  EXPECT_EQ(
      std::vector<uint8_t>({'x', 'i', 'F', 'd', ' ', 'S', 'e', 't', 0, 'r'}),
      DataBytes);

  Reader.setOffset(0);
  Writer.setOffset(0);
  ::memset(DataBytes.data(), 0, 10);
  EXPECT_NO_ERROR(Writer.writeArray(byteArray[0]));
  EXPECT_NO_ERROR(Reader.readArray(byteArray[1], byteArray[0].size()));
  EXPECT_EQ(byteArray[0], byteArray[1]);
  EXPECT_EQ(std::vector<uint8_t>({0, 0x32, 0x31, 0, 0, 0, 0, 0, 0, 0}),
            DataBytes);

  Reader.setOffset(0);
  Writer.setOffset(0);
  ::memset(DataBytes.data(), 0, 10);
  EXPECT_NO_ERROR(Writer.writeArray(intArray[0]));
  EXPECT_NO_ERROR(Reader.readArray(intArray[1], intArray[0].size()));
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
      F.block_size(), F.block_count(), F.layout(), F);

  // First write "Test Str" into the source stream.
  MutableBinaryByteStream SourceStream(SrcData);
  BinaryStreamWriter SourceWriter(SourceStream);
  EXPECT_NO_ERROR(SourceWriter.writeCString("Test Str"));
  EXPECT_EQ(SrcDataBytes, std::vector<uint8_t>(
                              {'T', 'e', 's', 't', ' ', 'S', 't', 'r', 0, 0}));

  // Then write the source stream into the dest stream.
  BinaryStreamWriter DestWriter(*DestStream);
  EXPECT_NO_ERROR(DestWriter.writeStreamRef(SourceStream));
  EXPECT_EQ(DestDataBytes, std::vector<uint8_t>(
                               {'s', 'e', 'T', ' ', 'S', 't', 't', 'r', 0, 0}));

  // Then read the string back out of the dest stream.
  StringRef Result;
  BinaryStreamReader DestReader(*DestStream);
  EXPECT_NO_ERROR(DestReader.readCString(Result));
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
      DestF.block_size(), DestF.block_count(), DestF.layout(), DestF);
  auto Src = WritableMappedBlockStream::createStream(
      SrcF.block_size(), SrcF.block_count(), SrcF.layout(), SrcF);

  // First write "Test Str" into the source stream.
  BinaryStreamWriter SourceWriter(*Src);
  EXPECT_NO_ERROR(SourceWriter.writeCString("Test Str"));
  EXPECT_EQ(SrcDataBytes, std::vector<uint8_t>(
                              {'e', 'T', 't', 't', ' ', 'S', 's', 'r', 0, 0}));

  // Then write the source stream into the dest stream.
  BinaryStreamWriter DestWriter(*Dest);
  EXPECT_NO_ERROR(DestWriter.writeStreamRef(*Src));
  EXPECT_EQ(DestDataBytes, std::vector<uint8_t>(
                               {'s', 'e', 'T', ' ', 'S', 't', 't', 'r', 0, 0}));

  // Then read the string back out of the dest stream.
  StringRef Result;
  BinaryStreamReader DestReader(*Dest);
  EXPECT_NO_ERROR(DestReader.readCString(Result));
  EXPECT_EQ(Result, "Test Str");
}

} // end anonymous namespace
