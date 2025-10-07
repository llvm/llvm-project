//===- SimplePackedSerializationTest.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for SimplePackedSerialization infrastructure.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/SimplePackedSerialization.h"

#include "SimplePackedSerializationTestUtils.h"
#include "gtest/gtest.h"

using namespace orc_rt;

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

TEST(SimplePackedSerializationTest, SizeTSerialization) {
  size_t V = 42;
  blobSerializationRoundTrip<SPSSize, size_t>(V);
}

TEST(SimplePackedSerializationTest, SequenceSerialization) {
  std::vector<int32_t> V({1, 2, -47, 139});
  blobSerializationRoundTrip<SPSSequence<int32_t>, std::vector<int32_t>>(V);
}

TEST(SimplePackedSerializationTest, ExecutorAddr) {
  int X = 42;
  auto A = ExecutorAddr::fromPtr(&X);
  blobSerializationRoundTrip<SPSExecutorAddr>(A);
}

TEST(SimplePackedSerializationTest, StringViewCharSequenceSerialization) {
  const char *HW = "Hello, world!";
  blobSerializationRoundTrip<SPSString, std::string_view>(std::string_view(HW));
}

TEST(SimplePackedSerializationTest, SpanSerialization) {
  const char Data[] = {3, 2, 1, 0, 1, 2, 3}; // Span should handle nulls.
  span<const char> OutS(Data, sizeof(Data));

  size_t Size = SPSArgList<SPSSequence<char>>::size(OutS);
  auto Buffer = std::make_unique<char[]>(Size);
  SPSOutputBuffer OB(Buffer.get(), Size);

  EXPECT_TRUE(SPSArgList<SPSSequence<char>>::serialize(OB, OutS));

  SPSInputBuffer IB(Buffer.get(), Size);

  span<const char> InS;

  EXPECT_TRUE(SPSArgList<SPSSequence<char>>::deserialize(IB, InS));

  // Check that the serialized and deserialized values match.
  EXPECT_EQ(InS.size(), OutS.size());
  EXPECT_EQ(memcmp(OutS.data(), InS.data(), InS.size()), 0);

  // Check that the span points directly to the input buffer.
  EXPECT_EQ(InS.data(), Buffer.get() + sizeof(uint64_t));
}

TEST(SimplePackedSerializationTest, StdTupleSerialization) {
  std::tuple<int32_t, std::string, bool> P(42, "foo", true);
  blobSerializationRoundTrip<SPSTuple<int32_t, SPSString, bool>>(P);
}

TEST(SimplePackedSerializationTest, StdPairSerialization) {
  std::pair<int32_t, std::string> P(42, "foo");
  blobSerializationRoundTrip<SPSTuple<int32_t, SPSString>,
                             std::pair<int32_t, std::string>>(P);
}

TEST(SimplePackedSerializationTest, StdOptionalNoValueSerialization) {
  std::optional<int64_t> NoValue;
  blobSerializationRoundTrip<SPSOptional<int64_t>>(NoValue);
}

TEST(SimplePackedSerializationTest, StdOptionalValueSerialization) {
  std::optional<int64_t> Value(42);
  blobSerializationRoundTrip<SPSOptional<int64_t>>(Value);
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

TEST(SimplePackedSerializationTest, SerializeErrorSuccess) {
  auto B = spsSerialize<SPSArgList<SPSError>>(
      SPSSerializableError(Error::success()));
  if (!B) {
    ADD_FAILURE() << "Unexpected failure to serialize error-success value";
    return;
  }
  SPSSerializableError SE;
  if (!spsDeserialize<SPSArgList<SPSError>>(*B, SE)) {
    ADD_FAILURE() << "Unexpected failure to deserialize error-success value";
    return;
  }

  auto E = SE.toError();
  EXPECT_FALSE(!!E); // Expect non-error, i.e. Error::success().
}

TEST(SimplePackedSerializationTest, SerializeErrorFailure) {
  auto B = spsSerialize<SPSArgList<SPSError>>(
      SPSSerializableError(make_error<StringError>("test error message")));
  if (!B) {
    ADD_FAILURE() << "Unexpected failure to serialize error-failure value";
    return;
  }
  SPSSerializableError SE;
  if (!spsDeserialize<SPSArgList<SPSError>>(*B, SE)) {
    ADD_FAILURE() << "Unexpected failure to deserialize error-failure value";
    return;
  }

  EXPECT_EQ(toString(SE.toError()), std::string("test error message"));
}

TEST(SimplePackedSerializationTest, SerializeExpectedSuccessViaExpected) {
  auto B = spsSerialize<SPSArgList<SPSExpected<uint32_t>>>(
      toSPSSerializableExpected(Expected<uint32_t>(42U)));
  if (!B) {
    ADD_FAILURE() << "Unexpected failure to serialize expected-success value";
    return;
  }
  SPSSerializableExpected<uint32_t> SE;
  if (!spsDeserialize<SPSArgList<SPSExpected<uint32_t>>>(*B, SE)) {
    ADD_FAILURE() << "Unexpected failure to deserialize expected-success value";
    return;
  }

  auto E = SE.toExpected();
  if (E)
    EXPECT_EQ(*E, 42U);
  else
    ADD_FAILURE() << "Unexpected failure value";
}

TEST(SimplePackedSerializationTest, SerializeExpectedSuccessViaValue) {
  auto B = spsSerialize<SPSArgList<SPSExpected<uint32_t>>>(
      toSPSSerializableExpected(uint32_t(42U)));
  if (!B) {
    ADD_FAILURE() << "Unexpected failure to serialize expected-success value";
    return;
  }
  SPSSerializableExpected<uint32_t> SE;
  if (!spsDeserialize<SPSArgList<SPSExpected<uint32_t>>>(*B, SE)) {
    ADD_FAILURE() << "Unexpected failure to deserialize expected-success value";
    return;
  }

  auto E = SE.toExpected();
  if (E)
    EXPECT_EQ(*E, 42U);
  else
    ADD_FAILURE() << "Unexpected failure value";
}

TEST(SimplePackedSerializationTest, SerializeExpectedFailure) {
  auto B = spsSerialize<SPSArgList<SPSExpected<uint32_t>>>(
      toSPSSerializableExpected<uint32_t>(
          make_error<StringError>("test error message")));
  if (!B) {
    ADD_FAILURE() << "Unexpected failure to serialize expected-failure value";
    return;
  }
  SPSSerializableExpected<uint32_t> SE;
  if (!spsDeserialize<SPSArgList<SPSExpected<uint32_t>>>(*B, SE)) {
    ADD_FAILURE() << "Unexpected failure to deserialize expected-failure value";
    return;
  }

  auto E = SE.toExpected();
  if (E)
    ADD_FAILURE() << "Unexpected failure value";
  else
    EXPECT_EQ(toString(E.takeError()), std::string("test error message"));
}
