//===-- RegisterValueTest.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/lldb-private-types.h"
#include "gtest/gtest.h"
#include <optional>

using namespace lldb_private;
using llvm::APInt;

TEST(RegisterValueTest, GetSet8) {
  RegisterValue R8(uint8_t(47));
  EXPECT_EQ(47u, R8.GetAsUInt8());
  R8 = uint8_t(42);
  EXPECT_EQ(42u, R8.GetAsUInt8());
  EXPECT_EQ(42u, R8.GetAsUInt16());
  EXPECT_EQ(42u, R8.GetAsUInt32());
  EXPECT_EQ(42u, R8.GetAsUInt64());
}

TEST(RegisterValueTest, GetScalarValue) {
  using RV = RegisterValue;
  const auto &Get = [](const RV &V) -> std::optional<Scalar> {
    Scalar S;
    if (V.GetScalarValue(S))
      return S;
    return std::nullopt;
  };
  EXPECT_EQ(Get(RV(uint8_t(47))), Scalar(47));
  EXPECT_EQ(Get(RV(uint16_t(4747))), Scalar(4747));
  EXPECT_EQ(Get(RV(uint32_t(47474242))), Scalar(47474242));
  EXPECT_EQ(Get(RV(uint64_t(4747424247474242))), Scalar(4747424247474242));
  EXPECT_EQ(Get(RV(APInt::getMaxValue(128))), Scalar(APInt::getMaxValue(128)));
  EXPECT_EQ(Get(RV(47.5f)), Scalar(47.5f));
  EXPECT_EQ(Get(RV(47.5)), Scalar(47.5));
  EXPECT_EQ(Get(RV(47.5L)), Scalar(47.5L));
  EXPECT_EQ(Get(RV({0xff, 0xee, 0xdd, 0xcc}, lldb::eByteOrderLittle)),
            Scalar(0xccddeeff));
  EXPECT_EQ(Get(RV({0xff, 0xee, 0xdd, 0xcc}, lldb::eByteOrderBig)),
            Scalar(0xffeeddcc));
  EXPECT_EQ(Get(RV({0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa, 0x99, 0x88, 0x77, 0x66,
                    0x55, 0x44, 0x33, 0x22, 0x11, 0x00},
                   lldb::eByteOrderLittle)),
            Scalar((APInt(128, 0x0011223344556677ull) << 64) |
                   APInt(128, 0x8899aabbccddeeff)));
  EXPECT_EQ(Get(RV({0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa, 0x99, 0x88, 0x77, 0x66,
                    0x55, 0x44, 0x33, 0x22, 0x11, 0x00},
                   lldb::eByteOrderBig)),
            Scalar((APInt(128, 0xffeeddccbbaa9988ull) << 64) |
                   APInt(128, 0x7766554433221100)));
}

void TestSetValueFromData(const Scalar &etalon, void *src, size_t src_byte_size,
                          const lldb::ByteOrder endianness) {
  RegisterInfo ri{"uint128_register",
                  nullptr,
                  static_cast<uint32_t>(src_byte_size),
                  0,
                  lldb::Encoding::eEncodingUint,
                  lldb::Format::eFormatDefault,
                  {0, 0, 0, LLDB_INVALID_REGNUM, 0},
                  nullptr,
                  nullptr,
                  nullptr};
  DataExtractor src_extractor(src, src_byte_size, endianness, 8);
  RegisterValue rv;
  EXPECT_TRUE(rv.SetValueFromData(ri, src_extractor, 0, false).Success());
  Scalar s;
  EXPECT_TRUE(rv.GetScalarValue(s));
  EXPECT_EQ(s, etalon);
}

static const Scalar etalon128(APInt(128, 0x0f0e0d0c0b0a0908ull) << 1 * 64 |
                              APInt(128, 0x0706050403020100ull) << 0 * 64);

// Test that the "RegisterValue::SetValueFromData" method works correctly
// with 128-bit little-endian data that represents an integer.
TEST(RegisterValueTest, SetValueFromData_128_le) {
  uint8_t src[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                   0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f};
  TestSetValueFromData(etalon128, src, 16, lldb::ByteOrder::eByteOrderLittle);
}

// Test that the "RegisterValue::SetValueFromData" method works correctly
// with 128-bit big-endian data that represents an integer.
TEST(RegisterValueTest, SetValueFromData_128_be) {
  uint8_t src[] = {0x0f, 0x0e, 0x0d, 0x0c, 0x0b, 0x0a, 0x09, 0x08,
                   0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x00};
  TestSetValueFromData(etalon128, src, 16, lldb::ByteOrder::eByteOrderBig);
}
