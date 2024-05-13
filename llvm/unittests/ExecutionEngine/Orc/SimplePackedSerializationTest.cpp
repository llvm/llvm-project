//===-------- SimplePackedSerializationTest.cpp - Test SPS scheme ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Shared/SimplePackedSerialization.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::orc::shared;

TEST(SimplePackedSerializationTest, SPSOutputBuffer) {
  constexpr unsigned NumBytes = 8;
  char Buffer[NumBytes];
  char Zero = 0;
  SPSOutputBuffer OB(Buffer, NumBytes);

  // Expect that we can write NumBytes of content.
  for (unsigned I = 0; I != NumBytes; ++I) {
    char C = I;
    EXPECT_TRUE(OB.write(&C, 1));
  }

  // Expect an error when we attempt to write an extra byte.
  EXPECT_FALSE(OB.write(&Zero, 1));

  // Check that the buffer contains the expected content.
  for (unsigned I = 0; I != NumBytes; ++I)
    EXPECT_EQ(Buffer[I], (char)I);
}

TEST(SimplePackedSerializationTest, SPSInputBuffer) {
  char Buffer[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07};
  SPSInputBuffer IB(Buffer, sizeof(Buffer));

  char C;
  for (unsigned I = 0; I != sizeof(Buffer); ++I) {
    EXPECT_TRUE(IB.read(&C, 1));
    EXPECT_EQ(C, (char)I);
  }

  EXPECT_FALSE(IB.read(&C, 1));
}

template <typename SPSTagT, typename T>
static void spsSerializationRoundTrip(const T &Value) {
  using BST = SPSSerializationTraits<SPSTagT, T>;

  size_t Size = BST::size(Value);
  auto Buffer = std::make_unique<char[]>(Size);
  SPSOutputBuffer OB(Buffer.get(), Size);

  EXPECT_TRUE(BST::serialize(OB, Value));

  SPSInputBuffer IB(Buffer.get(), Size);

  T DSValue;
  EXPECT_TRUE(BST::deserialize(IB, DSValue));

  EXPECT_EQ(Value, DSValue)
      << "Incorrect value after serialization/deserialization round-trip";
}

template <typename T> static void testFixedIntegralTypeSerialization() {
  spsSerializationRoundTrip<T, T>(0);
  spsSerializationRoundTrip<T, T>(static_cast<T>(1));
  if (std::is_signed_v<T>) {
    spsSerializationRoundTrip<T, T>(static_cast<T>(-1));
    spsSerializationRoundTrip<T, T>(std::numeric_limits<T>::min());
  }
  spsSerializationRoundTrip<T, T>(std::numeric_limits<T>::max());
}

TEST(SimplePackedSerializationTest, BoolSerialization) {
  spsSerializationRoundTrip<bool, bool>(true);
  spsSerializationRoundTrip<bool, bool>(false);
}

TEST(SimplePackedSerializationTest, CharSerialization) {
  spsSerializationRoundTrip<char, char>((char)0x00);
  spsSerializationRoundTrip<char, char>((char)0xAA);
  spsSerializationRoundTrip<char, char>((char)0xFF);
}

TEST(SimplePackedSerializationTest, Int8Serialization) {
  testFixedIntegralTypeSerialization<int8_t>();
}

TEST(SimplePackedSerializationTest, UInt8Serialization) {
  testFixedIntegralTypeSerialization<uint8_t>();
}

TEST(SimplePackedSerializationTest, Int16Serialization) {
  testFixedIntegralTypeSerialization<int16_t>();
}

TEST(SimplePackedSerializationTest, UInt16Serialization) {
  testFixedIntegralTypeSerialization<uint16_t>();
}

TEST(SimplePackedSerializationTest, Int32Serialization) {
  testFixedIntegralTypeSerialization<int32_t>();
}

TEST(SimplePackedSerializationTest, UInt32Serialization) {
  testFixedIntegralTypeSerialization<uint32_t>();
}

TEST(SimplePackedSerializationTest, Int64Serialization) {
  testFixedIntegralTypeSerialization<int64_t>();
}

TEST(SimplePackedSerializationTest, UInt64Serialization) {
  testFixedIntegralTypeSerialization<uint64_t>();
}

TEST(SimplePackedSerializationTest, SequenceSerialization) {
  std::vector<int32_t> V({1, 2, -47, 139});
  spsSerializationRoundTrip<SPSSequence<int32_t>>(V);
}

TEST(SimplePackedSerializationTest, StringViewCharSequenceSerialization) {
  const char *HW = "Hello, world!";
  spsSerializationRoundTrip<SPSString>(StringRef(HW));
}

TEST(SimplePackedSerializationTest, StdTupleSerialization) {
  std::tuple<int32_t, std::string, bool> P(42, "foo", true);
  spsSerializationRoundTrip<SPSTuple<int32_t, SPSString, bool>>(P);
}

TEST(SimplePackedSerializationTest, StdPairSerialization) {
  std::pair<int32_t, std::string> P(42, "foo");
  spsSerializationRoundTrip<SPSTuple<int32_t, SPSString>>(P);
}

TEST(SimplePackedSerializationTest, StdOptionalNoValueSerialization) {
  std::optional<int64_t> NoValue;
  spsSerializationRoundTrip<SPSOptional<int64_t>>(NoValue);
}

TEST(SimplePackedSerializationTest, StdOptionalValueSerialization) {
  std::optional<int64_t> Value(42);
  spsSerializationRoundTrip<SPSOptional<int64_t>>(Value);
}

TEST(SimplePackedSerializationTest, ArgListSerialization) {
  using BAL = SPSArgList<bool, int32_t, SPSString>;

  bool Arg1 = true;
  int32_t Arg2 = 42;
  std::string Arg3 = "foo";

  size_t Size = BAL::size(Arg1, Arg2, Arg3);
  auto Buffer = std::make_unique<char[]>(Size);
  SPSOutputBuffer OB(Buffer.get(), Size);

  EXPECT_TRUE(BAL::serialize(OB, Arg1, Arg2, Arg3));

  SPSInputBuffer IB(Buffer.get(), Size);

  bool ArgOut1;
  int32_t ArgOut2;
  std::string ArgOut3;

  EXPECT_TRUE(BAL::deserialize(IB, ArgOut1, ArgOut2, ArgOut3));

  EXPECT_EQ(Arg1, ArgOut1);
  EXPECT_EQ(Arg2, ArgOut2);
  EXPECT_EQ(Arg3, ArgOut3);
}

TEST(SimplePackedSerialization, StringMap) {
  StringMap<int32_t> M({{"A", 1}, {"B", 2}});
  spsSerializationRoundTrip<SPSSequence<SPSTuple<SPSString, int32_t>>>(M);
}

TEST(SimplePackedSerializationTest, ArrayRef) {
  constexpr unsigned BufferSize = 6 + 8; // "hello\0" + sizeof(uint64_t)
  ArrayRef<char> HelloOut = "hello";
  char Buffer[BufferSize];
  memset(Buffer, 0, BufferSize);

  SPSOutputBuffer OB(Buffer, BufferSize);
  EXPECT_TRUE(SPSArgList<SPSSequence<char>>::serialize(OB, HelloOut));

  ArrayRef<char> HelloIn;
  SPSInputBuffer IB(Buffer, BufferSize);
  EXPECT_TRUE(SPSArgList<SPSSequence<char>>::deserialize(IB, HelloIn));

  // Output should be copied to buffer.
  EXPECT_NE(HelloOut.data(), Buffer);

  // Input should reference buffer.
  EXPECT_LT(HelloIn.data() - Buffer, BufferSize);
}

TEST(SimplePackedSerializationTest, ArrayRefEmpty) {
  // Make sure that empty ArrayRefs serialize and deserialize as expected.
  // Empty ArrayRefs should not succeed even when the data field is null, and
  // should deserialize to a default-constructed ArrayRef, not a pointer into
  // the stream.
  constexpr unsigned BufferSize = sizeof(uint64_t);
  char Buffer[BufferSize];
  memset(Buffer, 0, BufferSize);

  ArrayRef<char> AOut;
  SPSOutputBuffer OB(Buffer, BufferSize);
  EXPECT_TRUE(SPSArgList<SPSSequence<char>>::serialize(OB, AOut));

  ArrayRef<char> AIn;
  SPSInputBuffer IB(Buffer, BufferSize);
  EXPECT_TRUE(SPSArgList<SPSSequence<char>>::deserialize(IB, AIn));

  EXPECT_EQ(AIn.data(), nullptr);
  EXPECT_EQ(AIn.size(), 0U);
}

TEST(SimplePackedSerializationTest, StringRefEmpty) {
  // Make sure that empty StringRefs serialize and deserialize as expected.
  // Empty StringRefs should not succeed even when the data field is null, and
  // should deserialize to a default-constructed StringRef, not a pointer into
  // the stream.
  constexpr unsigned BufferSize = sizeof(uint64_t);
  char Buffer[BufferSize];
  memset(Buffer, 0, BufferSize);

  StringRef SROut;
  SPSOutputBuffer OB(Buffer, BufferSize);
  EXPECT_TRUE(SPSArgList<SPSSequence<char>>::serialize(OB, SROut));

  StringRef SRIn;
  SPSInputBuffer IB(Buffer, BufferSize);
  EXPECT_TRUE(SPSArgList<SPSSequence<char>>::deserialize(IB, SRIn));

  EXPECT_EQ(SRIn.data(), nullptr);
  EXPECT_EQ(SRIn.size(), 0U);
}
