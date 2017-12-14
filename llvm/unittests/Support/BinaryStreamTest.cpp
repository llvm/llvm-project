//===- llvm/unittest/Support/BinaryStreamTest.cpp -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/BinaryByteStream.h"
#include "llvm/Support/BinaryItemStream.h"
#include "llvm/Support/BinaryStreamArray.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/BinaryStreamRef.h"
#include "llvm/Support/BinaryStreamWriter.h"
#include "llvm/Testing/Support/Error.h"

#include "gtest/gtest.h"


using namespace llvm;
using namespace llvm::support;

namespace {

class BrokenStream : public WritableBinaryStream {
public:
  BrokenStream(MutableArrayRef<uint8_t> Data, endianness Endian,
                      uint32_t Align)
      : Data(Data), PartitionIndex(alignDown(Data.size() / 2, Align)),
        Endian(Endian) {}

  endianness getEndian() const override { return Endian; }

  Error readBytes(uint32_t Offset, uint32_t Size,
                  ArrayRef<uint8_t> &Buffer) override {
    if (auto EC = checkOffsetForRead(Offset, Size))
      return EC;
    uint32_t S = startIndex(Offset);
    auto Ref = Data.drop_front(S);
    if (Ref.size() >= Size) {
      Buffer = Ref.take_front(Size);
      return Error::success();
    }

    uint32_t BytesLeft = Size - Ref.size();
    uint8_t *Ptr = Allocator.Allocate<uint8_t>(Size);
    ::memcpy(Ptr, Ref.data(), Ref.size());
    ::memcpy(Ptr + Ref.size(), Data.data(), BytesLeft);
    Buffer = makeArrayRef<uint8_t>(Ptr, Size);
    return Error::success();
  }

  Error readLongestContiguousChunk(uint32_t Offset,
                                   ArrayRef<uint8_t> &Buffer) override {
    if (auto EC = checkOffsetForRead(Offset, 1))
      return EC;
    uint32_t S = startIndex(Offset);
    Buffer = Data.drop_front(S);
    return Error::success();
  }

  uint32_t getLength() override { return Data.size(); }

  Error writeBytes(uint32_t Offset, ArrayRef<uint8_t> SrcData) override {
    if (auto EC = checkOffsetForWrite(Offset, SrcData.size()))
      return EC;
    if (SrcData.empty())
      return Error::success();

    uint32_t S = startIndex(Offset);
    MutableArrayRef<uint8_t> Ref(Data);
    Ref = Ref.drop_front(S);
    if (Ref.size() >= SrcData.size()) {
      ::memcpy(Ref.data(), SrcData.data(), SrcData.size());
      return Error::success();
    }

    uint32_t BytesLeft = SrcData.size() - Ref.size();
    ::memcpy(Ref.data(), SrcData.data(), Ref.size());
    ::memcpy(&Data[0], SrcData.data() + Ref.size(), BytesLeft);
    return Error::success();
  }
  Error commit() override { return Error::success(); }

private:
  uint32_t startIndex(uint32_t Offset) const {
    return (Offset + PartitionIndex) % Data.size();
  }

  uint32_t endIndex(uint32_t Offset, uint32_t Size) const {
    return (startIndex(Offset) + Size - 1) % Data.size();
  }

  // Buffer is organized like this:
  // -------------------------------------------------
  // | N/2 | N/2+1 | ... | N-1 | 0 | 1 | ... | N/2-1 |
  // -------------------------------------------------
  // So reads from the beginning actually come from the middle.
  MutableArrayRef<uint8_t> Data;
  uint32_t PartitionIndex = 0;
  endianness Endian;
  BumpPtrAllocator Allocator;
};

constexpr endianness Endians[] = {big, little, native};
constexpr uint32_t NumEndians = llvm::array_lengthof(Endians);
constexpr uint32_t NumStreams = 2 * NumEndians;

class BinaryStreamTest : public testing::Test {

public:
  BinaryStreamTest() {}

  void SetUp() override {
    Streams.clear();
    Streams.resize(NumStreams);
    for (uint32_t I = 0; I < NumStreams; ++I)
      Streams[I].IsContiguous = (I % 2 == 0);

    InputData.clear();
    OutputData.clear();
  }

protected:
  struct StreamPair {
    bool IsContiguous;
    std::unique_ptr<BinaryStream> Input;
    std::unique_ptr<WritableBinaryStream> Output;
  };

  void initializeInput(ArrayRef<uint8_t> Input, uint32_t Align) {
    InputData = Input;

    BrokenInputData.resize(InputData.size());
    if (!Input.empty()) {
      uint32_t PartitionIndex = alignDown(InputData.size() / 2, Align);
      uint32_t RightBytes = InputData.size() - PartitionIndex;
      uint32_t LeftBytes = PartitionIndex;
      if (RightBytes > 0)
        ::memcpy(&BrokenInputData[PartitionIndex], Input.data(), RightBytes);
      if (LeftBytes > 0)
        ::memcpy(&BrokenInputData[0], Input.data() + RightBytes, LeftBytes);
    }

    for (uint32_t I = 0; I < NumEndians; ++I) {
      auto InByteStream =
          llvm::make_unique<BinaryByteStream>(InputData, Endians[I]);
      auto InBrokenStream = llvm::make_unique<BrokenStream>(
          BrokenInputData, Endians[I], Align);

      Streams[I * 2].Input = std::move(InByteStream);
      Streams[I * 2 + 1].Input = std::move(InBrokenStream);
    }
  }

  void initializeOutput(uint32_t Size, uint32_t Align) {
    OutputData.resize(Size);
    BrokenOutputData.resize(Size);

    for (uint32_t I = 0; I < NumEndians; ++I) {
      Streams[I * 2].Output =
          llvm::make_unique<MutableBinaryByteStream>(OutputData, Endians[I]);
      Streams[I * 2 + 1].Output = llvm::make_unique<BrokenStream>(
          BrokenOutputData, Endians[I], Align);
    }
  }

  void initializeOutputFromInput(uint32_t Align) {
    for (uint32_t I = 0; I < NumEndians; ++I) {
      Streams[I * 2].Output =
          llvm::make_unique<MutableBinaryByteStream>(InputData, Endians[I]);
      Streams[I * 2 + 1].Output = llvm::make_unique<BrokenStream>(
          BrokenInputData, Endians[I], Align);
    }
  }

  void initializeInputFromOutput(uint32_t Align) {
    for (uint32_t I = 0; I < NumEndians; ++I) {
      Streams[I * 2].Input =
          llvm::make_unique<BinaryByteStream>(OutputData, Endians[I]);
      Streams[I * 2 + 1].Input = llvm::make_unique<BrokenStream>(
          BrokenOutputData, Endians[I], Align);
    }
  }

  std::vector<uint8_t> InputData;
  std::vector<uint8_t> BrokenInputData;

  std::vector<uint8_t> OutputData;
  std::vector<uint8_t> BrokenOutputData;

  std::vector<StreamPair> Streams;
};

// Tests that a we can read from a BinaryByteStream without a StreamReader.
TEST_F(BinaryStreamTest, BinaryByteStreamBounds) {
  std::vector<uint8_t> InputData = {1, 2, 3, 4, 5};
  initializeInput(InputData, 1);

  for (auto &Stream : Streams) {
    ArrayRef<uint8_t> Buffer;

    // 1. If the read fits it should work.
    ASSERT_EQ(InputData.size(), Stream.Input->getLength());
    ASSERT_THAT_ERROR(Stream.Input->readBytes(2, 1, Buffer), Succeeded());
    EXPECT_EQ(makeArrayRef(InputData).slice(2, 1), Buffer);
    ASSERT_THAT_ERROR(Stream.Input->readBytes(0, 4, Buffer), Succeeded());
    EXPECT_EQ(makeArrayRef(InputData).slice(0, 4), Buffer);

    // 2. Reading past the bounds of the input should fail.
    EXPECT_THAT_ERROR(Stream.Input->readBytes(4, 2, Buffer), Failed());
  }
}

TEST_F(BinaryStreamTest, StreamRefBounds) {
  std::vector<uint8_t> InputData = {1, 2, 3, 4, 5};
  initializeInput(InputData, 1);

  for (const auto &Stream : Streams) {
    ArrayRef<uint8_t> Buffer;
    BinaryStreamRef Ref(*Stream.Input);

    // Read 1 byte from offset 2 should work
    ASSERT_EQ(InputData.size(), Ref.getLength());
    ASSERT_THAT_ERROR(Ref.readBytes(2, 1, Buffer), Succeeded());
    EXPECT_EQ(makeArrayRef(InputData).slice(2, 1), Buffer);

    // Reading everything from offset 2 on.
    ASSERT_THAT_ERROR(Ref.readLongestContiguousChunk(2, Buffer), Succeeded());
    if (Stream.IsContiguous)
      EXPECT_EQ(makeArrayRef(InputData).slice(2), Buffer);
    else
      EXPECT_FALSE(Buffer.empty());

    // Reading 6 bytes from offset 0 is too big.
    EXPECT_THAT_ERROR(Ref.readBytes(0, 6, Buffer), Failed());
    EXPECT_THAT_ERROR(Ref.readLongestContiguousChunk(6, Buffer), Failed());

    // Reading 1 byte from offset 2 after dropping 1 byte is the same as reading
    // 1 byte from offset 3.
    Ref = Ref.drop_front(1);
    ASSERT_THAT_ERROR(Ref.readBytes(2, 1, Buffer), Succeeded());
    if (Stream.IsContiguous)
      EXPECT_EQ(makeArrayRef(InputData).slice(3, 1), Buffer);
    else
      EXPECT_FALSE(Buffer.empty());

    // Reading everything from offset 2 on after dropping 1 byte.
    ASSERT_THAT_ERROR(Ref.readLongestContiguousChunk(2, Buffer), Succeeded());
    if (Stream.IsContiguous)
      EXPECT_EQ(makeArrayRef(InputData).slice(3), Buffer);
    else
      EXPECT_FALSE(Buffer.empty());

    // Reading 2 bytes from offset 2 after dropping 2 bytes is the same as
    // reading 2 bytes from offset 4, and should fail.
    Ref = Ref.drop_front(1);
    EXPECT_THAT_ERROR(Ref.readBytes(2, 2, Buffer), Failed());

    // But if we read the longest contiguous chunk instead, we should still
    // get the 1 byte at the end.
    ASSERT_THAT_ERROR(Ref.readLongestContiguousChunk(2, Buffer), Succeeded());
    EXPECT_EQ(makeArrayRef(InputData).take_back(), Buffer);
  }
}

TEST_F(BinaryStreamTest, StreamRefDynamicSize) {
  StringRef Strings[] = {"1", "2", "3", "4"};
  AppendingBinaryByteStream Stream(support::little);

  BinaryStreamWriter Writer(Stream);
  BinaryStreamReader Reader(Stream);
  const uint8_t *Byte;
  StringRef Str;

  // When the stream is empty, it should report a 0 length and we should get an
  // error trying to read even 1 byte from it.
  BinaryStreamRef ConstRef(Stream);
  EXPECT_EQ(0U, ConstRef.getLength());
  EXPECT_THAT_ERROR(Reader.readObject(Byte), Failed());

  // But if we write to it, its size should increase and we should be able to
  // read not just a byte, but the string that was written.
  EXPECT_THAT_ERROR(Writer.writeCString(Strings[0]), Succeeded());
  EXPECT_EQ(2U, ConstRef.getLength());
  EXPECT_THAT_ERROR(Reader.readObject(Byte), Succeeded());

  Reader.setOffset(0);
  EXPECT_THAT_ERROR(Reader.readCString(Str), Succeeded());
  EXPECT_EQ(Str, Strings[0]);

  // If we drop some bytes from the front, we should still track the length as
  // the
  // underlying stream grows.
  BinaryStreamRef Dropped = ConstRef.drop_front(1);
  EXPECT_EQ(1U, Dropped.getLength());

  EXPECT_THAT_ERROR(Writer.writeCString(Strings[1]), Succeeded());
  EXPECT_EQ(4U, ConstRef.getLength());
  EXPECT_EQ(3U, Dropped.getLength());

  // If we drop zero bytes from the back, we should continue tracking the
  // length.
  Dropped = Dropped.drop_back(0);
  EXPECT_THAT_ERROR(Writer.writeCString(Strings[2]), Succeeded());
  EXPECT_EQ(6U, ConstRef.getLength());
  EXPECT_EQ(5U, Dropped.getLength());

  // If we drop non-zero bytes from the back, we should stop tracking the
  // length.
  Dropped = Dropped.drop_back(1);
  EXPECT_THAT_ERROR(Writer.writeCString(Strings[3]), Succeeded());
  EXPECT_EQ(8U, ConstRef.getLength());
  EXPECT_EQ(4U, Dropped.getLength());
}

TEST_F(BinaryStreamTest, DropOperations) {
  std::vector<uint8_t> InputData = {1, 2, 3, 4, 5, 4, 3, 2, 1};
  auto RefData = makeArrayRef(InputData);
  initializeInput(InputData, 1);

  ArrayRef<uint8_t> Result;
  BinaryStreamRef Original(InputData, support::little);
  ASSERT_EQ(InputData.size(), Original.getLength());

  EXPECT_THAT_ERROR(Original.readBytes(0, InputData.size(), Result),
                    Succeeded());
  EXPECT_EQ(RefData, Result);

  auto Dropped = Original.drop_front(2);
  EXPECT_THAT_ERROR(Dropped.readBytes(0, Dropped.getLength(), Result),
                    Succeeded());
  EXPECT_EQ(RefData.drop_front(2), Result);

  Dropped = Original.drop_back(2);
  EXPECT_THAT_ERROR(Dropped.readBytes(0, Dropped.getLength(), Result),
                    Succeeded());
  EXPECT_EQ(RefData.drop_back(2), Result);

  Dropped = Original.keep_front(2);
  EXPECT_THAT_ERROR(Dropped.readBytes(0, Dropped.getLength(), Result),
                    Succeeded());
  EXPECT_EQ(RefData.take_front(2), Result);

  Dropped = Original.keep_back(2);
  EXPECT_THAT_ERROR(Dropped.readBytes(0, Dropped.getLength(), Result),
                    Succeeded());
  EXPECT_EQ(RefData.take_back(2), Result);

  Dropped = Original.drop_symmetric(2);
  EXPECT_THAT_ERROR(Dropped.readBytes(0, Dropped.getLength(), Result),
                    Succeeded());
  EXPECT_EQ(RefData.drop_front(2).drop_back(2), Result);
}

// Test that we can write to a BinaryStream without a StreamWriter.
TEST_F(BinaryStreamTest, MutableBinaryByteStreamBounds) {
  std::vector<uint8_t> InputData = {'T', 'e', 's', 't', '\0'};
  initializeInput(InputData, 1);
  initializeOutput(InputData.size(), 1);

  // For every combination of input stream and output stream.
  for (auto &Stream : Streams) {
    ASSERT_EQ(InputData.size(), Stream.Input->getLength());

    // 1. Try two reads that are supposed to work.  One from offset 0, and one
    // from the middle.
    uint32_t Offsets[] = {0, 3};
    for (auto Offset : Offsets) {
      uint32_t ExpectedSize = Stream.Input->getLength() - Offset;

      // Read everything from Offset until the end of the input data.
      ArrayRef<uint8_t> Data;
      ASSERT_THAT_ERROR(Stream.Input->readBytes(Offset, ExpectedSize, Data),
                        Succeeded());
      ASSERT_EQ(ExpectedSize, Data.size());

      // Then write it to the destination.
      ASSERT_THAT_ERROR(Stream.Output->writeBytes(0, Data), Succeeded());

      // Then we read back what we wrote, it should match the corresponding
      // slice of the original input data.
      ArrayRef<uint8_t> Data2;
      ASSERT_THAT_ERROR(Stream.Output->readBytes(Offset, ExpectedSize, Data2),
                        Succeeded());
      EXPECT_EQ(makeArrayRef(InputData).drop_front(Offset), Data2);
    }

    std::vector<uint8_t> BigData = {0, 1, 2, 3, 4};
    // 2. If the write is too big, it should fail.
    EXPECT_THAT_ERROR(Stream.Output->writeBytes(3, BigData), Failed());
  }
}

TEST_F(BinaryStreamTest, AppendingStream) {
  AppendingBinaryByteStream Stream(llvm::support::little);
  EXPECT_EQ(0U, Stream.getLength());

  std::vector<uint8_t> InputData = {'T', 'e', 's', 't', 'T', 'e', 's', 't'};
  auto Test = makeArrayRef(InputData).take_front(4);
  // Writing past the end of the stream is an error.
  EXPECT_THAT_ERROR(Stream.writeBytes(4, Test), Failed());

  // Writing exactly at the end of the stream is ok.
  EXPECT_THAT_ERROR(Stream.writeBytes(0, Test), Succeeded());
  EXPECT_EQ(Test, Stream.data());

  // And now that the end of the stream is where we couldn't write before, now
  // we can write.
  EXPECT_THAT_ERROR(Stream.writeBytes(4, Test), Succeeded());
  EXPECT_EQ(MutableArrayRef<uint8_t>(InputData), Stream.data());
}

// Test that FixedStreamArray works correctly.
TEST_F(BinaryStreamTest, FixedStreamArray) {
  std::vector<uint32_t> Ints = {90823, 12908, 109823, 209823};
  ArrayRef<uint8_t> IntBytes(reinterpret_cast<uint8_t *>(Ints.data()),
                             Ints.size() * sizeof(uint32_t));

  initializeInput(IntBytes, alignof(uint32_t));

  for (auto &Stream : Streams) {
    ASSERT_EQ(InputData.size(), Stream.Input->getLength());

    FixedStreamArray<uint32_t> Array(*Stream.Input);
    auto Iter = Array.begin();
    ASSERT_EQ(Ints[0], *Iter++);
    ASSERT_EQ(Ints[1], *Iter++);
    ASSERT_EQ(Ints[2], *Iter++);
    ASSERT_EQ(Ints[3], *Iter++);
    ASSERT_EQ(Array.end(), Iter);
  }
}

// Ensure FixedStreamArrayIterator::operator-> works.
// Added for coverage of r302257.
TEST_F(BinaryStreamTest, FixedStreamArrayIteratorArrow) {
  std::vector<std::pair<uint32_t, uint32_t>> Pairs = {{867, 5309}, {555, 1212}};
  ArrayRef<uint8_t> PairBytes(reinterpret_cast<uint8_t *>(Pairs.data()),
    Pairs.size() * sizeof(Pairs[0]));

  initializeInput(PairBytes, alignof(uint32_t));

  for (auto &Stream : Streams) {
    ASSERT_EQ(InputData.size(), Stream.Input->getLength());

    const FixedStreamArray<std::pair<uint32_t, uint32_t>> Array(*Stream.Input);
    auto Iter = Array.begin();
    ASSERT_EQ(Pairs[0].first, Iter->first);
    ASSERT_EQ(Pairs[0].second, Iter->second);
    ++Iter;
    ASSERT_EQ(Pairs[1].first, Iter->first);
    ASSERT_EQ(Pairs[1].second, Iter->second);
    ++Iter;
    ASSERT_EQ(Array.end(), Iter);
  }
}

// Test that VarStreamArray works correctly.
TEST_F(BinaryStreamTest, VarStreamArray) {
  StringLiteral Strings("1. Test2. Longer Test3. Really Long Test4. Super "
                        "Extra Longest Test Of All");
  ArrayRef<uint8_t> StringBytes(
      reinterpret_cast<const uint8_t *>(Strings.data()), Strings.size());
  initializeInput(StringBytes, 1);

  struct StringExtractor {
  public:
    Error operator()(BinaryStreamRef Stream, uint32_t &Len, StringRef &Item) {
      if (Index == 0)
        Len = strlen("1. Test");
      else if (Index == 1)
        Len = strlen("2. Longer Test");
      else if (Index == 2)
        Len = strlen("3. Really Long Test");
      else
        Len = strlen("4. Super Extra Longest Test Of All");
      ArrayRef<uint8_t> Bytes;
      if (auto EC = Stream.readBytes(0, Len, Bytes))
        return EC;
      Item =
          StringRef(reinterpret_cast<const char *>(Bytes.data()), Bytes.size());
      ++Index;
      return Error::success();
    }

    uint32_t Index = 0;
  };

  for (auto &Stream : Streams) {
    VarStreamArray<StringRef, StringExtractor> Array(*Stream.Input);
    auto Iter = Array.begin();
    ASSERT_EQ("1. Test", *Iter++);
    ASSERT_EQ("2. Longer Test", *Iter++);
    ASSERT_EQ("3. Really Long Test", *Iter++);
    ASSERT_EQ("4. Super Extra Longest Test Of All", *Iter++);
    ASSERT_EQ(Array.end(), Iter);
  }
}

TEST_F(BinaryStreamTest, StreamReaderBounds) {
  std::vector<uint8_t> Bytes;

  initializeInput(Bytes, 1);
  for (auto &Stream : Streams) {
    StringRef S;
    BinaryStreamReader Reader(*Stream.Input);
    EXPECT_EQ(0U, Reader.bytesRemaining());
    EXPECT_THAT_ERROR(Reader.readFixedString(S, 1), Failed());
  }

  Bytes.resize(5);
  initializeInput(Bytes, 1);
  for (auto &Stream : Streams) {
    StringRef S;
    BinaryStreamReader Reader(*Stream.Input);
    EXPECT_EQ(Bytes.size(), Reader.bytesRemaining());
    EXPECT_THAT_ERROR(Reader.readFixedString(S, 5), Succeeded());
    EXPECT_THAT_ERROR(Reader.readFixedString(S, 6), Failed());
  }
}

TEST_F(BinaryStreamTest, StreamReaderIntegers) {
  support::ulittle64_t Little{908234};
  support::ubig32_t Big{28907823};
  short NS = 2897;
  int NI = -89723;
  unsigned long NUL = 902309023UL;
  constexpr uint32_t Size =
      sizeof(Little) + sizeof(Big) + sizeof(NS) + sizeof(NI) + sizeof(NUL);

  initializeOutput(Size, alignof(support::ulittle64_t));
  initializeInputFromOutput(alignof(support::ulittle64_t));

  for (auto &Stream : Streams) {
    BinaryStreamWriter Writer(*Stream.Output);
    ASSERT_THAT_ERROR(Writer.writeObject(Little), Succeeded());
    ASSERT_THAT_ERROR(Writer.writeObject(Big), Succeeded());
    ASSERT_THAT_ERROR(Writer.writeInteger(NS), Succeeded());
    ASSERT_THAT_ERROR(Writer.writeInteger(NI), Succeeded());
    ASSERT_THAT_ERROR(Writer.writeInteger(NUL), Succeeded());

    const support::ulittle64_t *Little2;
    const support::ubig32_t *Big2;
    short NS2;
    int NI2;
    unsigned long NUL2;

    // 1. Reading fields individually.
    BinaryStreamReader Reader(*Stream.Input);
    ASSERT_THAT_ERROR(Reader.readObject(Little2), Succeeded());
    ASSERT_THAT_ERROR(Reader.readObject(Big2), Succeeded());
    ASSERT_THAT_ERROR(Reader.readInteger(NS2), Succeeded());
    ASSERT_THAT_ERROR(Reader.readInteger(NI2), Succeeded());
    ASSERT_THAT_ERROR(Reader.readInteger(NUL2), Succeeded());
    ASSERT_EQ(0U, Reader.bytesRemaining());

    EXPECT_EQ(Little, *Little2);
    EXPECT_EQ(Big, *Big2);
    EXPECT_EQ(NS, NS2);
    EXPECT_EQ(NI, NI2);
    EXPECT_EQ(NUL, NUL2);
  }
}

TEST_F(BinaryStreamTest, StreamReaderIntegerArray) {
  // 1. Arrays of integers
  std::vector<int> Ints = {1, 2, 3, 4, 5};
  ArrayRef<uint8_t> IntBytes(reinterpret_cast<uint8_t *>(&Ints[0]),
                             Ints.size() * sizeof(int));

  initializeInput(IntBytes, alignof(int));
  for (auto &Stream : Streams) {
    BinaryStreamReader Reader(*Stream.Input);
    ArrayRef<int> IntsRef;
    ASSERT_THAT_ERROR(Reader.readArray(IntsRef, Ints.size()), Succeeded());
    ASSERT_EQ(0U, Reader.bytesRemaining());
    EXPECT_EQ(makeArrayRef(Ints), IntsRef);

    Reader.setOffset(0);
    FixedStreamArray<int> FixedIntsRef;
    ASSERT_THAT_ERROR(Reader.readArray(FixedIntsRef, Ints.size()), Succeeded());
    ASSERT_EQ(0U, Reader.bytesRemaining());
    ASSERT_EQ(Ints, std::vector<int>(FixedIntsRef.begin(), FixedIntsRef.end()));
  }
}

TEST_F(BinaryStreamTest, StreamReaderEnum) {
  enum class MyEnum : int64_t { Foo = -10, Bar = 0, Baz = 10 };

  std::vector<MyEnum> Enums = {MyEnum::Bar, MyEnum::Baz, MyEnum::Foo};

  initializeOutput(Enums.size() * sizeof(MyEnum), alignof(MyEnum));
  initializeInputFromOutput(alignof(MyEnum));
  for (auto &Stream : Streams) {
    BinaryStreamWriter Writer(*Stream.Output);
    for (auto Value : Enums)
      ASSERT_THAT_ERROR(Writer.writeEnum(Value), Succeeded());

    BinaryStreamReader Reader(*Stream.Input);

    FixedStreamArray<MyEnum> FSA;

    for (size_t I = 0; I < Enums.size(); ++I) {
      MyEnum Value;
      ASSERT_THAT_ERROR(Reader.readEnum(Value), Succeeded());
      EXPECT_EQ(Enums[I], Value);
    }
    ASSERT_EQ(0U, Reader.bytesRemaining());
  }
}

TEST_F(BinaryStreamTest, StreamReaderObject) {
  struct Foo {
    int X;
    double Y;
    char Z;

    bool operator==(const Foo &Other) const {
      return X == Other.X && Y == Other.Y && Z == Other.Z;
    }
  };

  std::vector<Foo> Foos;
  Foos.push_back({-42, 42.42, 42});
  Foos.push_back({100, 3.1415, static_cast<char>(-89)});
  Foos.push_back({200, 2.718, static_cast<char>(-12) });

  const uint8_t *Bytes = reinterpret_cast<const uint8_t *>(&Foos[0]);

  initializeInput(makeArrayRef(Bytes, 3 * sizeof(Foo)), alignof(Foo));

  for (auto &Stream : Streams) {
    // 1. Reading object pointers.
    BinaryStreamReader Reader(*Stream.Input);
    const Foo *FPtrOut = nullptr;
    const Foo *GPtrOut = nullptr;
    const Foo *HPtrOut = nullptr;
    ASSERT_THAT_ERROR(Reader.readObject(FPtrOut), Succeeded());
    ASSERT_THAT_ERROR(Reader.readObject(GPtrOut), Succeeded());
    ASSERT_THAT_ERROR(Reader.readObject(HPtrOut), Succeeded());
    EXPECT_EQ(0U, Reader.bytesRemaining());
    EXPECT_EQ(Foos[0], *FPtrOut);
    EXPECT_EQ(Foos[1], *GPtrOut);
    EXPECT_EQ(Foos[2], *HPtrOut);
  }
}

TEST_F(BinaryStreamTest, StreamReaderStrings) {
  std::vector<uint8_t> Bytes = {'O',  'n', 'e', '\0', 'T', 'w', 'o',
                                '\0', 'T', 'h', 'r',  'e', 'e', '\0',
                                'F',  'o', 'u', 'r',  '\0'};
  initializeInput(Bytes, 1);

  for (auto &Stream : Streams) {
    BinaryStreamReader Reader(*Stream.Input);

    StringRef S1;
    StringRef S2;
    StringRef S3;
    StringRef S4;
    ASSERT_THAT_ERROR(Reader.readCString(S1), Succeeded());
    ASSERT_THAT_ERROR(Reader.readCString(S2), Succeeded());
    ASSERT_THAT_ERROR(Reader.readCString(S3), Succeeded());
    ASSERT_THAT_ERROR(Reader.readCString(S4), Succeeded());
    ASSERT_EQ(0U, Reader.bytesRemaining());

    EXPECT_EQ("One", S1);
    EXPECT_EQ("Two", S2);
    EXPECT_EQ("Three", S3);
    EXPECT_EQ("Four", S4);

    S1 = S2 = S3 = S4 = "";
    Reader.setOffset(0);
    ASSERT_THAT_ERROR(Reader.readFixedString(S1, 3), Succeeded());
    ASSERT_THAT_ERROR(Reader.skip(1), Succeeded());
    ASSERT_THAT_ERROR(Reader.readFixedString(S2, 3), Succeeded());
    ASSERT_THAT_ERROR(Reader.skip(1), Succeeded());
    ASSERT_THAT_ERROR(Reader.readFixedString(S3, 5), Succeeded());
    ASSERT_THAT_ERROR(Reader.skip(1), Succeeded());
    ASSERT_THAT_ERROR(Reader.readFixedString(S4, 4), Succeeded());
    ASSERT_THAT_ERROR(Reader.skip(1), Succeeded());
    ASSERT_EQ(0U, Reader.bytesRemaining());

    EXPECT_EQ("One", S1);
    EXPECT_EQ("Two", S2);
    EXPECT_EQ("Three", S3);
    EXPECT_EQ("Four", S4);
  }
}

TEST_F(BinaryStreamTest, StreamWriterBounds) {
  initializeOutput(5, 1);

  for (auto &Stream : Streams) {
    BinaryStreamWriter Writer(*Stream.Output);

    // 1. Can write a string that exactly fills the buffer.
    EXPECT_EQ(5U, Writer.bytesRemaining());
    EXPECT_THAT_ERROR(Writer.writeFixedString("abcde"), Succeeded());
    EXPECT_EQ(0U, Writer.bytesRemaining());

    // 2. Can write an empty string even when you're full
    EXPECT_THAT_ERROR(Writer.writeFixedString(""), Succeeded());
    EXPECT_THAT_ERROR(Writer.writeFixedString("a"), Failed());

    // 3. Can't write a string that is one character too long.
    Writer.setOffset(0);
    EXPECT_THAT_ERROR(Writer.writeFixedString("abcdef"), Failed());
  }
}

TEST_F(BinaryStreamTest, StreamWriterIntegerArrays) {
  // 3. Arrays of integers
  std::vector<int> SourceInts = {1, 2, 3, 4, 5};
  ArrayRef<uint8_t> SourceBytes(reinterpret_cast<uint8_t *>(&SourceInts[0]),
                                SourceInts.size() * sizeof(int));

  initializeInput(SourceBytes, alignof(int));
  initializeOutputFromInput(alignof(int));

  for (auto &Stream : Streams) {
    BinaryStreamReader Reader(*Stream.Input);
    BinaryStreamWriter Writer(*Stream.Output);
    ArrayRef<int> Ints;
    ArrayRef<int> Ints2;
    // First read them, then write them, then read them back.
    ASSERT_THAT_ERROR(Reader.readArray(Ints, SourceInts.size()), Succeeded());
    ASSERT_THAT_ERROR(Writer.writeArray(Ints), Succeeded());

    BinaryStreamReader ReaderBacker(*Stream.Output);
    ASSERT_THAT_ERROR(ReaderBacker.readArray(Ints2, SourceInts.size()),
                      Succeeded());

    EXPECT_EQ(makeArrayRef(SourceInts), Ints2);
  }
}

TEST_F(BinaryStreamTest, StringWriterStrings) {
  StringRef Strings[] = {"First", "Second", "Third", "Fourth"};

  size_t Length = 0;
  for (auto S : Strings)
    Length += S.size() + 1;
  initializeOutput(Length, 1);
  initializeInputFromOutput(1);

  for (auto &Stream : Streams) {
    BinaryStreamWriter Writer(*Stream.Output);
    for (auto S : Strings)
      ASSERT_THAT_ERROR(Writer.writeCString(S), Succeeded());
    std::vector<StringRef> InStrings;
    BinaryStreamReader Reader(*Stream.Input);
    while (!Reader.empty()) {
      StringRef S;
      ASSERT_THAT_ERROR(Reader.readCString(S), Succeeded());
      InStrings.push_back(S);
    }
    EXPECT_EQ(makeArrayRef(Strings), makeArrayRef(InStrings));
  }
}

TEST_F(BinaryStreamTest, StreamWriterAppend) {
  StringRef Strings[] = {"First", "Second", "Third", "Fourth"};
  AppendingBinaryByteStream Stream(support::little);
  BinaryStreamWriter Writer(Stream);

  for (auto &Str : Strings) {
    EXPECT_THAT_ERROR(Writer.writeCString(Str), Succeeded());
  }

  BinaryStreamReader Reader(Stream);
  for (auto &Str : Strings) {
    StringRef S;
    EXPECT_THAT_ERROR(Reader.readCString(S), Succeeded());
    EXPECT_EQ(Str, S);
  }
}
}

namespace {
struct BinaryItemStreamObject {
  explicit BinaryItemStreamObject(ArrayRef<uint8_t> Bytes) : Bytes(Bytes) {}

  ArrayRef<uint8_t> Bytes;
};
}

namespace llvm {
template <> struct BinaryItemTraits<BinaryItemStreamObject> {
  static size_t length(const BinaryItemStreamObject &Item) {
    return Item.Bytes.size();
  }

  static ArrayRef<uint8_t> bytes(const BinaryItemStreamObject &Item) {
    return Item.Bytes;
  }
};
}

namespace {

TEST_F(BinaryStreamTest, BinaryItemStream) {
  std::vector<BinaryItemStreamObject> Objects;

  struct Foo {
    int X;
    double Y;
  };
  std::vector<Foo> Foos = {{1, 1.0}, {2, 2.0}, {3, 3.0}};
  BumpPtrAllocator Allocator;
  for (const auto &F : Foos) {
    uint8_t *Ptr = static_cast<uint8_t *>(Allocator.Allocate(sizeof(Foo),
                                                             alignof(Foo)));
    MutableArrayRef<uint8_t> Buffer(Ptr, sizeof(Foo));
    MutableBinaryByteStream Stream(Buffer, llvm::support::big);
    BinaryStreamWriter Writer(Stream);
    ASSERT_THAT_ERROR(Writer.writeObject(F), Succeeded());
    Objects.push_back(BinaryItemStreamObject(Buffer));
  }

  BinaryItemStream<BinaryItemStreamObject> ItemStream(big);
  ItemStream.setItems(Objects);
  BinaryStreamReader Reader(ItemStream);

  for (const auto &F : Foos) {
    const Foo *F2;
    ASSERT_THAT_ERROR(Reader.readObject(F2), Succeeded());

    EXPECT_EQ(F.X, F2->X);
    EXPECT_DOUBLE_EQ(F.Y, F2->Y);
  }
}

} // end anonymous namespace
