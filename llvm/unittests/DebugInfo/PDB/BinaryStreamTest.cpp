//===- llvm/unittest/Support/BinaryStreamTest.cpp -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/MSF/BinaryByteStream.h"
#include "llvm/DebugInfo/MSF/BinaryItemStream.h"
#include "llvm/DebugInfo/MSF/BinaryStreamArray.h"
#include "llvm/DebugInfo/MSF/BinaryStreamReader.h"
#include "llvm/DebugInfo/MSF/BinaryStreamRef.h"
#include "llvm/DebugInfo/MSF/BinaryStreamWriter.h"
#include "gtest/gtest.h"

#include <unordered_map>

using namespace llvm;
using namespace llvm::support;

#define EXPECT_NO_ERROR(Err)                                                   \
  {                                                                            \
    auto E = Err;                                                              \
    EXPECT_FALSE(static_cast<bool>(E));                                        \
    if (E)                                                                     \
      consumeError(std::move(E));                                              \
  }

#define ASSERT_NO_ERROR(Err)                                                   \
  {                                                                            \
    auto E = Err;                                                              \
    ASSERT_FALSE(static_cast<bool>(E));                                        \
    if (E)                                                                     \
      consumeError(std::move(E));                                              \
  }

#define EXPECT_ERROR(Err)                                                      \
  {                                                                            \
    auto E = Err;                                                              \
    EXPECT_TRUE(static_cast<bool>(E));                                         \
    if (E)                                                                     \
      consumeError(std::move(E));                                              \
  }

namespace {

class DiscontiguousStream : public WritableBinaryStream {
public:
  explicit DiscontiguousStream(uint32_t Size = 0) : PartitionIndex(Size / 2) {
    Data.resize(Size);
  }

  endianness getEndian() const override { return little; }

  Error readBytes(uint32_t Offset, uint32_t Size,
                  ArrayRef<uint8_t> &Buffer) override {
    if (Offset + Size > Data.size())
      return errorCodeToError(make_error_code(std::errc::no_buffer_space));
    uint32_t S = startIndex(Offset);
    auto Ref = makeArrayRef(Data).drop_front(S);
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
    if (Offset >= Data.size())
      return errorCodeToError(make_error_code(std::errc::no_buffer_space));
    uint32_t S = startIndex(Offset);
    Buffer = makeArrayRef(Data).drop_front(S);
    return Error::success();
  }

  uint32_t getLength() override { return Data.size(); }

  Error writeBytes(uint32_t Offset, ArrayRef<uint8_t> SrcData) override {
    if (Offset + SrcData.size() > Data.size())
      return errorCodeToError(make_error_code(std::errc::no_buffer_space));
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

  uint32_t PartitionIndex = 0;
  // Buffer is organized like this:
  // -------------------------------------------------
  // | N/2 | N/2+1 | ... | N-1 | 0 | 1 | ... | N-2-1 |
  // -------------------------------------------------
  // So reads from the beginning actually come from the middle.
  std::vector<uint8_t> Data;
  BumpPtrAllocator Allocator;
};

class BinaryStreamTest : public testing::Test {
public:
  BinaryStreamTest() {}

  void SetUp() override {
    InputData.clear();
    OutputData.clear();
    InputByteStream = BinaryByteStream();
    InputBrokenStream = DiscontiguousStream();
    OutputByteStream = MutableBinaryByteStream();
    OutputBrokenStream = DiscontiguousStream();
  }

protected:
  void initialize(ArrayRef<uint8_t> Input, uint32_t OutputSize) {
    InputData = Input;

    InputByteStream = BinaryByteStream(InputData, little);
    InputBrokenStream = DiscontiguousStream(InputData.size());
    consumeError(InputBrokenStream.writeBytes(0, Input));

    OutputData.resize(OutputSize);
    OutputByteStream = MutableBinaryByteStream(OutputData, little);
    OutputBrokenStream = DiscontiguousStream(OutputSize);

    InputStreams.push_back(&InputByteStream);
    InputStreams.push_back(&InputBrokenStream);
    OutputStreams.push_back(&OutputByteStream);
    OutputStreams.push_back(&OutputBrokenStream);
  }

  void initialize(uint32_t OutputSize) {
    OutputData.resize(OutputSize);
    OutputByteStream = MutableBinaryByteStream(OutputData, little);
    OutputBrokenStream = DiscontiguousStream(OutputSize);
    OutputStreams.push_back(&OutputByteStream);
    OutputStreams.push_back(&OutputBrokenStream);

    InputByteStream = BinaryByteStream(OutputData, little);
    InputBrokenStream = DiscontiguousStream(OutputData.size());
  }

  std::vector<uint8_t> InputData;
  std::vector<uint8_t> OutputData;

  BinaryByteStream InputByteStream;
  DiscontiguousStream InputBrokenStream;

  MutableBinaryByteStream OutputByteStream;
  DiscontiguousStream OutputBrokenStream;

  std::vector<BinaryStream *> InputStreams;
  std::vector<WritableBinaryStream *> OutputStreams;
};

// Tests that a we can read from a BinaryByteStream without a StreamReader.
TEST_F(BinaryStreamTest, BinaryByteStreamProperties) {
  std::vector<uint8_t> InputData = {1, 2, 3, 4, 5};
  initialize(InputData, InputData.size());

  for (auto Stream : InputStreams) {
    ArrayRef<uint8_t> Buffer;

    // 1. If the read fits it should work.
    ASSERT_EQ(InputData.size(), Stream->getLength());
    ASSERT_NO_ERROR(Stream->readBytes(2, 1, Buffer));
    EXPECT_EQ(makeArrayRef(InputData).slice(2, 1), Buffer);
    ASSERT_NO_ERROR(Stream->readBytes(0, 4, Buffer));
    EXPECT_EQ(makeArrayRef(InputData).slice(0, 4), Buffer);

    // 2. Reading past the bounds of the input should fail.
    EXPECT_ERROR(Stream->readBytes(4, 2, Buffer));
  }
}

// Test that we can write to a BinaryStream without a StreamWriter.
TEST_F(BinaryStreamTest, MutableBinaryByteStreamProperties) {
  std::vector<uint8_t> InputData = {'T', 'e', 's', 't', '\0'};
  initialize(InputData, InputData.size());
  ASSERT_EQ(2U, InputStreams.size());
  ASSERT_EQ(2U, OutputStreams.size());

  // For every combination of input stream and output stream.
  for (auto IS : InputStreams) {
    MutableArrayRef<uint8_t> Buffer;
    ASSERT_EQ(InputData.size(), IS->getLength());

    for (auto OS : OutputStreams) {

      // 1. Try two reads that are supposed to work.  One from offset 0, and one
      // from the middle.
      uint32_t Offsets[] = {0, 3};
      for (auto Offset : Offsets) {
        uint32_t ExpectedSize = IS->getLength() - Offset;

        // Read everything from Offset until the end of the input data.
        ArrayRef<uint8_t> Data;
        ASSERT_NO_ERROR(IS->readBytes(Offset, ExpectedSize, Data));
        ASSERT_EQ(ExpectedSize, Data.size());

        // Then write it to the destination.
        ASSERT_NO_ERROR(OS->writeBytes(0, Data));

        // Then we read back what we wrote, it should match the corresponding
        // slice
        // of the original input data.
        ArrayRef<uint8_t> Data2;
        ASSERT_NO_ERROR(OS->readBytes(Offset, ExpectedSize, Data2));
        EXPECT_EQ(makeArrayRef(InputData).drop_front(Offset), Data2);
      }

      std::vector<uint8_t> BigData = {0, 1, 2, 3, 4};
      // 2. If the write is too big, it should fail.
      EXPECT_ERROR(OS->writeBytes(3, BigData));
    }
  }
}

// Test that FixedStreamArray works correctly.
TEST_F(BinaryStreamTest, FixedStreamArray) {
  std::vector<uint32_t> Ints = {90823, 12908, 109823, 209823};
  ArrayRef<uint8_t> IntBytes(reinterpret_cast<uint8_t *>(Ints.data()),
                             Ints.size() * sizeof(uint32_t));

  initialize(IntBytes, 0);
  ASSERT_EQ(2U, InputStreams.size());

  for (auto IS : InputStreams) {
    MutableArrayRef<uint8_t> Buffer;
    ASSERT_EQ(InputData.size(), IS->getLength());

    FixedStreamArray<uint32_t> Array(*IS);
    auto Iter = Array.begin();
    ASSERT_EQ(Ints[0], *Iter++);
    ASSERT_EQ(Ints[1], *Iter++);
    ASSERT_EQ(Ints[2], *Iter++);
    ASSERT_EQ(Ints[3], *Iter++);
    ASSERT_EQ(Array.end(), Iter);
  }
}

// Test that VarStreamArray works correctly.
TEST_F(BinaryStreamTest, VarStreamArray) {
  StringLiteral Strings("1. Test2. Longer Test3. Really Long Test4. Super "
                        "Extra Longest Test Of All");
  ArrayRef<uint8_t> StringBytes(
      reinterpret_cast<const uint8_t *>(Strings.data()), Strings.size());
  initialize(StringBytes, 0);

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

  private:
    uint32_t Index = 0;
  };

  for (auto IS : InputStreams) {
    VarStreamArray<StringRef, StringExtractor> Array(*IS);
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

  initialize(Bytes, 0);
  for (auto IS : InputStreams) {
    StringRef S;
    BinaryStreamReader Reader(*IS);
    EXPECT_EQ(0U, Reader.bytesRemaining());
    EXPECT_ERROR(Reader.readFixedString(S, 1));
  }

  Bytes.resize(5);
  initialize(Bytes, 0);
  for (auto IS : InputStreams) {
    StringRef S;
    BinaryStreamReader Reader(*IS);
    EXPECT_EQ(Bytes.size(), Reader.bytesRemaining());
    EXPECT_NO_ERROR(Reader.readFixedString(S, 5));
    EXPECT_ERROR(Reader.readFixedString(S, 6));
  }
}

TEST_F(BinaryStreamTest, DISABLED_StreamReaderIntegers) {
  support::ulittle64_t Little{908234};
  support::ubig32_t Big{28907823};
  short NS = 2897;
  int NI = -89723;
  unsigned long NUL = 902309023UL;
  constexpr uint32_t Size =
      sizeof(Little) + sizeof(Big) + sizeof(NS) + sizeof(NI) + sizeof(NUL);
  std::vector<uint8_t> Bytes(Size);
  uint8_t *Ptr = &Bytes[0];
  memcpy(Ptr, &Little, sizeof(Little));
  Ptr += sizeof(Little);
  memcpy(Ptr, &Big, sizeof(Big));
  Ptr += sizeof(Big);
  memcpy(Ptr, &NS, sizeof(NS));
  Ptr += sizeof(NS);
  memcpy(Ptr, &NI, sizeof(NI));
  Ptr += sizeof(NI);
  memcpy(Ptr, &NUL, sizeof(NUL));
  Ptr += sizeof(NUL);

  initialize(Bytes, 0);
  for (auto IS : InputStreams) {
    const support::ulittle64_t *Little2;
    const support::ubig32_t *Big2;
    short NS2;
    int NI2;
    unsigned long NUL2;

    // 1. Reading fields individually.
    BinaryStreamReader Reader(*IS);
    ASSERT_NO_ERROR(Reader.readObject(Little2));
    ASSERT_NO_ERROR(Reader.readObject(Big2));
    ASSERT_NO_ERROR(Reader.readInteger(NS2));
    ASSERT_NO_ERROR(Reader.readInteger(NI2));
    ASSERT_NO_ERROR(Reader.readInteger(NUL2));
    ASSERT_EQ(0U, Reader.bytesRemaining());

    EXPECT_EQ(Little, *Little2);
    EXPECT_EQ(Big, *Big2);
    EXPECT_EQ(NS, NS2);
    EXPECT_EQ(NI, NI2);
    EXPECT_EQ(NUL, NUL2);

    // 2. Reading with explicit endianness.
    Reader.setOffset(0);
    const ulittle64_t *Little3;
    const ubig32_t *Big3;
    ASSERT_NO_ERROR(Reader.readObject(Little3));
    ASSERT_NO_ERROR(Reader.readObject(Big3));
    EXPECT_EQ(Little, *Little3);
    EXPECT_EQ(Big, *Big3);
  }
}

TEST_F(BinaryStreamTest, StreamReaderIntegerArray) {
  // 1. Arrays of integers
  std::vector<int> Ints = {1, 2, 3, 4, 5};
  ArrayRef<uint8_t> IntBytes(reinterpret_cast<uint8_t *>(&Ints[0]),
                             Ints.size() * sizeof(int));
  initialize(IntBytes, 0);
  for (auto IS : InputStreams) {
    BinaryStreamReader Reader(*IS);
    ArrayRef<int> IntsRef;
    ASSERT_NO_ERROR(Reader.readArray(IntsRef, Ints.size()));
    ASSERT_EQ(0U, Reader.bytesRemaining());
    EXPECT_EQ(makeArrayRef(Ints), IntsRef);

    Reader.setOffset(0);
    FixedStreamArray<int> FixedIntsRef;
    ASSERT_NO_ERROR(Reader.readArray(FixedIntsRef, Ints.size()));
    ASSERT_EQ(0U, Reader.bytesRemaining());
    ASSERT_EQ(Ints, std::vector<int>(FixedIntsRef.begin(), FixedIntsRef.end()));
  }
}

TEST_F(BinaryStreamTest, DISABLED_StreamReaderEnum) {
  enum class MyEnum : int64_t { Foo = -10, Bar = 0, Baz = 10 };

  std::vector<MyEnum> Enums = {MyEnum::Bar, MyEnum::Baz, MyEnum::Foo};

  ArrayRef<uint8_t> Bytes(reinterpret_cast<const uint8_t *>(&Enums[0]),
                          sizeof(MyEnum) * Enums.size());

  initialize(Bytes, 0);
  for (auto IS : InputStreams) {
    BinaryStreamReader Reader(*IS);

    MyEnum V1;
    MyEnum V2;
    MyEnum V3;
    ArrayRef<MyEnum> Array;
    FixedStreamArray<MyEnum> FSA;

    ASSERT_NO_ERROR(Reader.readEnum(V1));
    ASSERT_NO_ERROR(Reader.readEnum(V2));
    ASSERT_NO_ERROR(Reader.readEnum(V3));
    ASSERT_EQ(0U, Reader.bytesRemaining());

    EXPECT_EQ(MyEnum::Bar, V1);
    EXPECT_EQ(MyEnum::Baz, V2);
    EXPECT_EQ(MyEnum::Foo, V3);

    Reader.setOffset(0);
    ASSERT_NO_ERROR(Reader.readArray(Array, 3));
    EXPECT_EQ(makeArrayRef(Enums), Array);

    Reader.setOffset(0);
    ASSERT_NO_ERROR(Reader.readArray(FSA, 3));
    EXPECT_EQ(Enums, std::vector<MyEnum>(FSA.begin(), FSA.end()));
  }
}

TEST_F(BinaryStreamTest, StreamReaderObject) {
  struct Foo {
    int X;
    double Y;
    char Z;
  };

  std::vector<Foo> Foos;
  Foos.push_back({-42, 42.42, 42});
  Foos.push_back({100, 3.1415, -89});

  const uint8_t *Bytes = reinterpret_cast<const uint8_t *>(&Foos[0]);

  initialize(makeArrayRef(Bytes, 2 * sizeof(Foo)), 0);

  for (auto IS : InputStreams) {
    // 1. Reading object pointers.
    BinaryStreamReader Reader(*IS);
    const Foo *FPtrOut = nullptr;
    const Foo *GPtrOut = nullptr;
    ASSERT_NO_ERROR(Reader.readObject(FPtrOut));
    ASSERT_NO_ERROR(Reader.readObject(GPtrOut));
    EXPECT_EQ(0U, Reader.bytesRemaining());
    EXPECT_EQ(0, ::memcmp(&Foos[0], FPtrOut, sizeof(Foo)));
    EXPECT_EQ(0, ::memcmp(&Foos[1], GPtrOut, sizeof(Foo)));
  }
}

TEST_F(BinaryStreamTest, StreamReaderStrings) {
  std::vector<uint8_t> Bytes = {'O',  'n', 'e', '\0', 'T', 'w', 'o',
                                '\0', 'T', 'h', 'r',  'e', 'e', '\0',
                                'F',  'o', 'u', 'r',  '\0'};
  initialize(Bytes, 0);

  for (auto IS : InputStreams) {
    BinaryStreamReader Reader(*IS);

    StringRef S1;
    StringRef S2;
    StringRef S3;
    StringRef S4;
    ASSERT_NO_ERROR(Reader.readCString(S1));
    ASSERT_NO_ERROR(Reader.readCString(S2));
    ASSERT_NO_ERROR(Reader.readCString(S3));
    ASSERT_NO_ERROR(Reader.readCString(S4));
    ASSERT_EQ(0U, Reader.bytesRemaining());

    EXPECT_EQ("One", S1);
    EXPECT_EQ("Two", S2);
    EXPECT_EQ("Three", S3);
    EXPECT_EQ("Four", S4);

    S1 = S2 = S3 = S4 = "";
    Reader.setOffset(0);
    ASSERT_NO_ERROR(Reader.readFixedString(S1, 3));
    ASSERT_NO_ERROR(Reader.skip(1));
    ASSERT_NO_ERROR(Reader.readFixedString(S2, 3));
    ASSERT_NO_ERROR(Reader.skip(1));
    ASSERT_NO_ERROR(Reader.readFixedString(S3, 5));
    ASSERT_NO_ERROR(Reader.skip(1));
    ASSERT_NO_ERROR(Reader.readFixedString(S4, 4));
    ASSERT_NO_ERROR(Reader.skip(1));
    ASSERT_EQ(0U, Reader.bytesRemaining());

    EXPECT_EQ("One", S1);
    EXPECT_EQ("Two", S2);
    EXPECT_EQ("Three", S3);
    EXPECT_EQ("Four", S4);
  }
}

TEST_F(BinaryStreamTest, StreamWriterBounds) {
  initialize(5);

  for (auto OS : OutputStreams) {
    BinaryStreamWriter Writer(*OS);

    // 1. Can write a string that exactly fills the buffer.
    EXPECT_EQ(5U, Writer.bytesRemaining());
    EXPECT_NO_ERROR(Writer.writeFixedString("abcde"));
    EXPECT_EQ(0U, Writer.bytesRemaining());

    // 2. Can write an empty string even when you're full
    EXPECT_NO_ERROR(Writer.writeFixedString(""));
    EXPECT_ERROR(Writer.writeFixedString("a"));

    // 3. Can't write a string that is one character too long.
    Writer.setOffset(0);
    EXPECT_ERROR(Writer.writeFixedString("abcdef"));
  }
}

TEST_F(BinaryStreamTest, StreamWriterIntegers) {
  support::ulittle64_t Little{908234};
  support::ubig32_t Big{28907823};
  short NS = 2897;
  int NI = -89723;
  unsigned long NUL = 902309023UL;
  constexpr uint32_t Size =
      sizeof(Little) + sizeof(Big) + sizeof(NS) + sizeof(NI) + sizeof(NUL);

  initialize(Size);

  for (auto OS : OutputStreams) {
    BinaryStreamWriter Writer(*OS);

    // 1. Writing fields individually.
    ASSERT_NO_ERROR(Writer.writeObject(Little));
    ASSERT_NO_ERROR(Writer.writeObject(Big));
    ASSERT_NO_ERROR(Writer.writeInteger(NS));
    ASSERT_NO_ERROR(Writer.writeInteger(NI));
    ASSERT_NO_ERROR(Writer.writeInteger(NUL));
    ASSERT_EQ(0U, Writer.bytesRemaining());

    // Read them back in and confirm they're correct.
    const ulittle64_t *Little2;
    const ubig32_t *Big2;
    short NS2;
    int NI2;
    unsigned long NUL2;
    BinaryStreamReader Reader(*OS);
    ASSERT_NO_ERROR(Reader.readObject(Little2));
    ASSERT_NO_ERROR(Reader.readObject(Big2));
    ASSERT_NO_ERROR(Reader.readInteger(NS2));
    ASSERT_NO_ERROR(Reader.readInteger(NI2));
    ASSERT_NO_ERROR(Reader.readInteger(NUL2));
    EXPECT_EQ(Little, *Little2);
    EXPECT_EQ(Big, *Big2);
    EXPECT_EQ(NS, NS2);
    EXPECT_EQ(NI, NI2);
    EXPECT_EQ(NUL, NUL2);
  }
}

TEST_F(BinaryStreamTest, StreamWriterIntegerArrays) {
  // 3. Arrays of integers
  std::vector<int> SourceInts = {1, 2, 3, 4, 5};
  ArrayRef<uint8_t> SourceBytes(reinterpret_cast<uint8_t *>(&SourceInts[0]),
                                SourceInts.size() * sizeof(int));

  initialize(SourceBytes, SourceBytes.size());

  for (auto IS : InputStreams) {
    for (auto OS : OutputStreams) {
      BinaryStreamReader Reader(*IS);
      BinaryStreamWriter Writer(*OS);
      ArrayRef<int> Ints;
      ArrayRef<int> Ints2;
      // First read them, then write them, then read them back.
      ASSERT_NO_ERROR(Reader.readArray(Ints, SourceInts.size()));
      ASSERT_NO_ERROR(Writer.writeArray(Ints));

      BinaryStreamReader ReaderBacker(*OS);
      ASSERT_NO_ERROR(ReaderBacker.readArray(Ints2, SourceInts.size()));

      EXPECT_EQ(makeArrayRef(SourceInts), Ints2);
    }
  }
}

TEST_F(BinaryStreamTest, DISABLED_StreamWriterEnum) {
  enum class MyEnum : int64_t { Foo = -10, Bar = 0, Baz = 10 };

  std::vector<MyEnum> Expected = {MyEnum::Bar, MyEnum::Foo, MyEnum::Baz};

  initialize(Expected.size() * sizeof(MyEnum));

  for (auto OS : OutputStreams) {
    BinaryStreamWriter Writer(*OS);
    ArrayRef<MyEnum> Enums;
    ArrayRef<MyEnum> Enums2;

    // First read them, then write them, then read them back.
    for (auto ME : Expected)
      ASSERT_NO_ERROR(Writer.writeEnum(ME));

    ArrayRef<MyEnum> Array;
    BinaryStreamReader Reader(*OS);
    ASSERT_NO_ERROR(Reader.readArray(Array, Expected.size()));

    EXPECT_EQ(makeArrayRef(Expected), Array);
  }
}

TEST_F(BinaryStreamTest, StringWriterStrings) {
  StringRef Strings[] = {"First", "Second", "Third", "Fourth"};

  size_t Length = 0;
  for (auto S : Strings)
    Length += S.size() + 1;
  initialize(Length);

  for (auto OS : OutputStreams) {
    BinaryStreamWriter Writer(*OS);
    for (auto S : Strings)
      ASSERT_NO_ERROR(Writer.writeCString(S));

    for (auto IS : InputStreams) {
      std::vector<StringRef> InStrings;
      BinaryStreamReader Reader(*IS);
      while (!Reader.empty()) {
        StringRef S;
        ASSERT_NO_ERROR(Reader.readCString(S));
        InStrings.push_back(S);
      }
      EXPECT_EQ(makeArrayRef(Strings), makeArrayRef(InStrings));
    }
  }
}

TEST_F(BinaryStreamTest, StreamReaderIntegersVariadic) {
  uint8_t A = 201;
  int8_t A2 = -92;
  uint16_t B = 20823;
  int16_t B2 = -20823;
  uint32_t C = 8978251;
  int32_t C2 = -8978251;
  uint64_t D = 90278410232ULL;
  int64_t D2 = -90278410232LL;

  initialize(2 * (sizeof(A) + sizeof(B) + sizeof(C) + sizeof(D)));

  for (auto OS : OutputStreams) {
    BinaryStreamWriter Writer(*OS);
    ASSERT_NO_ERROR(Writer.writeIntegers(A, A2, B, B2, C, C2, D, D2));

    for (auto IS : InputStreams) {
      BinaryStreamReader Reader(*IS);
      uint8_t AX;
      int8_t AX2;
      uint16_t BX;
      int16_t BX2;
      uint32_t CX;
      int32_t CX2;
      uint64_t DX;
      int64_t DX2;

      ASSERT_NO_ERROR(Reader.readIntegers(AX, AX2, BX, BX2, CX, CX2, DX, DX2));
      EXPECT_EQ(A, AX);
      EXPECT_EQ(A2, AX2);
      EXPECT_EQ(B, BX);
      EXPECT_EQ(B2, BX2);
      EXPECT_EQ(C, CX);
      EXPECT_EQ(C2, CX2);
      EXPECT_EQ(D, DX);
      EXPECT_EQ(D2, DX2);
    }
  }
}
}

namespace {
struct BinaryItemStreamObject {
  BinaryItemStreamObject(int X, float Y) : X(X), Y(Y) {}

  int X;
  float Y;
};
}

namespace llvm {
template <> struct BinaryItemTraits<std::unique_ptr<BinaryItemStreamObject>> {
  size_t length(const std::unique_ptr<BinaryItemStreamObject> &Item) {
    size_t S = sizeof(Item->X);
    S += sizeof(Item->Y);
    return S;
  }

  ArrayRef<uint8_t> bytes(const std::unique_ptr<BinaryItemStreamObject> &Item) {
    // In practice we probably would use a more cheaply serializable type,
    // or at the very least not allocate every single time.  This is just
    // for illustration and testing though.
    size_t Size = length(Item);
    uint8_t *Buffer = Alloc.Allocate<uint8_t>(Size);
    MutableBinaryByteStream Stream(MutableArrayRef<uint8_t>(Buffer, Size),
                                   little);
    BinaryStreamWriter Writer(Stream);
    consumeError(Writer.writeInteger(Item->X));
    consumeError(Writer.writeObject(Item->Y));
    return makeArrayRef(Buffer, Size);
  }

private:
  BumpPtrAllocator Alloc;
};
}

namespace {

TEST_F(BinaryStreamTest, BinaryItemStream) {
  // Note that this is a vector of pointers, so individual records do not live
  // contiguously in memory.
  std::vector<std::unique_ptr<BinaryItemStreamObject>> Objects;
  Objects.push_back(llvm::make_unique<BinaryItemStreamObject>(1, 1.0));
  Objects.push_back(llvm::make_unique<BinaryItemStreamObject>(2, 2.0));
  Objects.push_back(llvm::make_unique<BinaryItemStreamObject>(3, 3.0));

  BinaryItemStream<std::unique_ptr<BinaryItemStreamObject>> ItemStream(little);
  ItemStream.setItems(Objects);
  BinaryStreamReader Reader(ItemStream);

  for (int I = 0; I < 3; ++I) {
    int X;
    const float *Y;
    ASSERT_NO_ERROR(Reader.readInteger(X));
    ASSERT_NO_ERROR(Reader.readObject(Y));

    EXPECT_EQ(Objects[I]->X, X);
    EXPECT_DOUBLE_EQ(Objects[I]->Y, *Y);
  }
}

} // end anonymous namespace
