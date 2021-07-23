//===-------- SimplePackedSerializationTest.cpp - Test SPS scheme ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Shared/SimplePackedSerialization.h"
#include "llvm/ExecutionEngine/Orc/LLVMSPSSerializers.h"
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
static void blobSerializationRoundTrip(const T &Value) {
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
  blobSerializationRoundTrip<T, T>(0);
  blobSerializationRoundTrip<T, T>(static_cast<T>(1));
  if (std::is_signed<T>::value) {
    blobSerializationRoundTrip<T, T>(static_cast<T>(-1));
    blobSerializationRoundTrip<T, T>(std::numeric_limits<T>::min());
  }
  blobSerializationRoundTrip<T, T>(std::numeric_limits<T>::max());
}

TEST(SimplePackedSerializationTest, BoolSerialization) {
  blobSerializationRoundTrip<bool, bool>(true);
  blobSerializationRoundTrip<bool, bool>(false);
}

TEST(SimplePackedSerializationTest, CharSerialization) {
  blobSerializationRoundTrip<char, char>((char)0x00);
  blobSerializationRoundTrip<char, char>((char)0xAA);
  blobSerializationRoundTrip<char, char>((char)0xFF);
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
  blobSerializationRoundTrip<SPSSequence<int32_t>, std::vector<int32_t>>(V);
}

TEST(SimplePackedSerializationTest, StringViewCharSequenceSerialization) {
  const char *HW = "Hello, world!";
  blobSerializationRoundTrip<SPSString, StringRef>(StringRef(HW));
}

TEST(SimplePackedSerializationTest, StdPairSerialization) {
  std::pair<int32_t, std::string> P(42, "foo");
  blobSerializationRoundTrip<SPSTuple<int32_t, SPSString>,
                             std::pair<int32_t, std::string>>(P);
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
  blobSerializationRoundTrip<SPSSequence<SPSTuple<SPSString, int32_t>>,
                             StringMap<int32_t>>(M);
}
