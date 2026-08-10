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
                          const lldb::ByteOrder endianness,
                          const RegisterValue::Type register_value_type) {
  RegisterInfo ri{"test",
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
  EXPECT_EQ(rv.GetType(), register_value_type);
  EXPECT_EQ(s, etalon);
}

static const Scalar etalon7(APInt(32, 0x0000007F));

TEST(RegisterValueTest, SetValueFromData_7_le) {
  uint8_t src[] = {0x7F};
  TestSetValueFromData(etalon7, src, 1, lldb::ByteOrder::eByteOrderLittle,
                       RegisterValue::eTypeUInt8);
}

TEST(RegisterValueTest, SetValueFromData_7_be) {
  uint8_t src[] = {0x7F};
  TestSetValueFromData(etalon7, src, 1, lldb::ByteOrder::eByteOrderBig,
                       RegisterValue::eTypeUInt8);
}

static const Scalar etalon8(APInt(32, 0x000000FE));

TEST(RegisterValueTest, SetValueFromData_8_le) {
  uint8_t src[] = {0xFE};
  TestSetValueFromData(etalon8, src, 1, lldb::ByteOrder::eByteOrderLittle,
                       RegisterValue::eTypeUInt8);
}

TEST(RegisterValueTest, SetValueFromData_8_be) {
  uint8_t src[] = {0xFE};
  TestSetValueFromData(etalon8, src, 1, lldb::ByteOrder::eByteOrderBig,
                       RegisterValue::eTypeUInt8);
}

static const Scalar etalon9(APInt(32, 0x000001FE));

TEST(RegisterValueTest, SetValueFromData_9_le) {
  uint8_t src[] = {0xFE, 0x01};
  TestSetValueFromData(etalon9, src, 2, lldb::ByteOrder::eByteOrderLittle,
                       RegisterValue::eTypeUInt16);
}

TEST(RegisterValueTest, SetValueFromData_9_be) {
  uint8_t src[] = {0x01, 0xFE};
  TestSetValueFromData(etalon9, src, 2, lldb::ByteOrder::eByteOrderBig,
                       RegisterValue::eTypeUInt16);
}

static const Scalar etalon15(APInt(32, 0x00007FED));

TEST(RegisterValueTest, SetValueFromData_15_le) {
  uint8_t src[] = {0xED, 0x7F};
  TestSetValueFromData(etalon15, src, 2, lldb::ByteOrder::eByteOrderLittle,
                       RegisterValue::eTypeUInt16);
}

TEST(RegisterValueTest, SetValueFromData_15_be) {
  uint8_t src[] = {0x7F, 0xED};
  TestSetValueFromData(etalon15, src, 2, lldb::ByteOrder::eByteOrderBig,
                       RegisterValue::eTypeUInt16);
}

static const Scalar etalon16(APInt(32, 0x0000FEDC));

TEST(RegisterValueTest, SetValueFromData_16_le) {
  uint8_t src[] = {0xDC, 0xFE};
  TestSetValueFromData(etalon16, src, 2, lldb::ByteOrder::eByteOrderLittle,
                       RegisterValue::eTypeUInt16);
}

TEST(RegisterValueTest, SetValueFromData_16_be) {
  uint8_t src[] = {0xFE, 0xDC};
  TestSetValueFromData(etalon16, src, 2, lldb::ByteOrder::eByteOrderBig,
                       RegisterValue::eTypeUInt16);
}

static const Scalar etalon17(APInt(32, 0x0001FEDC));

TEST(RegisterValueTest, SetValueFromData_17_le) {
  uint8_t src[] = {0xDC, 0xFE, 0x01};
  TestSetValueFromData(etalon17, src, 3, lldb::ByteOrder::eByteOrderLittle,
                       RegisterValue::eTypeUInt32);
}

TEST(RegisterValueTest, SetValueFromData_17_be) {
  uint8_t src[] = {0x01, 0xFE, 0xDC};
  TestSetValueFromData(etalon17, src, 3, lldb::ByteOrder::eByteOrderBig,
                       RegisterValue::eTypeUInt32);
}

static const Scalar etalon24(APInt(32, 0x00FEDCBA));

TEST(RegisterValueTest, SetValueFromData_24_le) {
  uint8_t src[] = {0xBA, 0xDC, 0xFE};
  TestSetValueFromData(etalon24, src, 3, lldb::ByteOrder::eByteOrderLittle,
                       RegisterValue::eTypeUInt32);
}

TEST(RegisterValueTest, SetValueFromData_24_be) {
  uint8_t src[] = {0xFE, 0xDC, 0xBA};
  TestSetValueFromData(etalon24, src, 3, lldb::ByteOrder::eByteOrderBig,
                       RegisterValue::eTypeUInt32);
}

static const Scalar etalon31(APInt(32, 0x7EDCBA98));

TEST(RegisterValueTest, SetValueFromData_31_le) {
  uint8_t src[] = {0x98, 0xBA, 0xDC, 0x7E};
  TestSetValueFromData(etalon31, src, 4, lldb::ByteOrder::eByteOrderLittle,
                       RegisterValue::eTypeUInt32);
}

TEST(RegisterValueTest, SetValueFromData_31_be) {
  uint8_t src[] = {0x7E, 0xDC, 0xBA, 0x98};
  TestSetValueFromData(etalon31, src, 4, lldb::ByteOrder::eByteOrderBig,
                       RegisterValue::eTypeUInt32);
}

static const Scalar etalon32(APInt(32, 0xFEDCBA98));

TEST(RegisterValueTest, SetValueFromData_32_le) {
  uint8_t src[] = {0x98, 0xBA, 0xDC, 0xFE};
  TestSetValueFromData(etalon32, src, 4, lldb::ByteOrder::eByteOrderLittle,
                       RegisterValue::eTypeUInt32);
}

TEST(RegisterValueTest, SetValueFromData_32_be) {
  uint8_t src[] = {0xFE, 0xDC, 0xBA, 0x98};
  TestSetValueFromData(etalon32, src, 4, lldb::ByteOrder::eByteOrderBig,
                       RegisterValue::eTypeUInt32);
}

static const Scalar etalon33(APInt(64, 0x00000001FEDCBA98));

TEST(RegisterValueTest, SetValueFromData_33_le) {
  uint8_t src[] = {0x98, 0xBA, 0xDC, 0xFE, 0x01};
  TestSetValueFromData(etalon33, src, 5, lldb::ByteOrder::eByteOrderLittle,
                       RegisterValue::eTypeUInt64);
}

TEST(RegisterValueTest, SetValueFromData_33_be) {
  uint8_t src[] = {0x01, 0xFE, 0xDC, 0xBA, 0x98};
  TestSetValueFromData(etalon33, src, 5, lldb::ByteOrder::eByteOrderBig,
                       RegisterValue::eTypeUInt64);
}

static const Scalar etalon40(APInt(64, 0x000000FEDCBA9876));

TEST(RegisterValueTest, SetValueFromData_40_le) {
  uint8_t src[] = {0x76, 0x98, 0xBA, 0xDC, 0xFE};
  TestSetValueFromData(etalon40, src, 5, lldb::ByteOrder::eByteOrderLittle,
                       RegisterValue::eTypeUInt64);
}

TEST(RegisterValueTest, SetValueFromData_40_be) {
  uint8_t src[] = {0xFE, 0xDC, 0xBA, 0x98, 0x76};
  TestSetValueFromData(etalon40, src, 5, lldb::ByteOrder::eByteOrderBig,
                       RegisterValue::eTypeUInt64);
}

static const Scalar etalon63(APInt(64, 0x7EDCBA9876543210));

TEST(RegisterValueTest, SetValueFromData_63_le) {
  uint8_t src[] = {0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0x7E};
  TestSetValueFromData(etalon63, src, 8, lldb::ByteOrder::eByteOrderLittle,
                       RegisterValue::eTypeUInt64);
}

TEST(RegisterValueTest, SetValueFromData_63_be) {
  uint8_t src[] = {0x7E, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10};
  TestSetValueFromData(etalon63, src, 8, lldb::ByteOrder::eByteOrderBig,
                       RegisterValue::eTypeUInt64);
}

static const Scalar etalon64(APInt(64, 0xFEDCBA9876543210));

TEST(RegisterValueTest, SetValueFromData_64_le) {
  uint8_t src[] = {0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE};
  TestSetValueFromData(etalon64, src, 8, lldb::ByteOrder::eByteOrderLittle,
                       RegisterValue::eTypeUInt64);
}

TEST(RegisterValueTest, SetValueFromData_64_be) {
  uint8_t src[] = {0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10};
  TestSetValueFromData(etalon64, src, 8, lldb::ByteOrder::eByteOrderBig,
                       RegisterValue::eTypeUInt64);
}

static const Scalar etalon65(APInt(72, 0x0000000000000001ull) << 1 * 64 |
                             APInt(72, 0x0706050403020100ull) << 0 * 64);

TEST(RegisterValueTest, SetValueFromData_65_le) {
  uint8_t src[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x01};
  TestSetValueFromData(etalon65, src, 9, lldb::ByteOrder::eByteOrderLittle,
                       RegisterValue::eTypeUIntN);
}

TEST(RegisterValueTest, SetValueFromData_65_be) {
  uint8_t src[] = {0x01, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x00};
  TestSetValueFromData(etalon65, src, 9, lldb::ByteOrder::eByteOrderBig,
                       RegisterValue::eTypeUIntN);
}

static const Scalar etalon127(APInt(128, 0x7f0e0d0c0b0a0908ull) << 1 * 64 |
                              APInt(128, 0x0706050403020100ull) << 0 * 64);

TEST(RegisterValueTest, SetValueFromData_127_le) {
  uint8_t src[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                   0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x7f};
  TestSetValueFromData(etalon127, src, 16, lldb::ByteOrder::eByteOrderLittle,
                       RegisterValue::eTypeUIntN);
}

TEST(RegisterValueTest, SetValueFromData_127_be) {
  uint8_t src[] = {0x7f, 0x0e, 0x0d, 0x0c, 0x0b, 0x0a, 0x09, 0x08,
                   0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x00};
  TestSetValueFromData(etalon127, src, 16, lldb::ByteOrder::eByteOrderBig,
                       RegisterValue::eTypeUIntN);
}

static const Scalar etalon128(APInt(128, 0x0f0e0d0c0b0a0908ull) << 1 * 64 |
                              APInt(128, 0x0706050403020100ull) << 0 * 64);

TEST(RegisterValueTest, SetValueFromData_128_le) {
  uint8_t src[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                   0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f};
  TestSetValueFromData(etalon128, src, 16, lldb::ByteOrder::eByteOrderLittle,
                       RegisterValue::eTypeUIntN);
}

TEST(RegisterValueTest, SetValueFromData_128_be) {
  uint8_t src[] = {0x0f, 0x0e, 0x0d, 0x0c, 0x0b, 0x0a, 0x09, 0x08,
                   0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x00};
  TestSetValueFromData(etalon128, src, 16, lldb::ByteOrder::eByteOrderBig,
                       RegisterValue::eTypeUIntN);
}

static const Scalar etalon256(APInt(256, 0x1f1e1d1c1b1a1918ull) << 3 * 64 |
                              APInt(256, 0x1716151413121110ull) << 2 * 64 |
                              APInt(256, 0x0f0e0d0c0b0a0908ull) << 1 * 64 |
                              APInt(256, 0x0706050403020100ull) << 0 * 64);

TEST(RegisterValueTest, SetValueFromData_256_le) {
  uint8_t src[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                   0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
                   0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
                   0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f};
  TestSetValueFromData(etalon256, src, 32, lldb::ByteOrder::eByteOrderLittle,
                       RegisterValue::eTypeUIntN);
}

TEST(RegisterValueTest, SetValueFromData_256_be) {
  uint8_t src[] = {0x1f, 0x1e, 0x1d, 0x1c, 0x1b, 0x1a, 0x19, 0x18,
                   0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11, 0x10,
                   0x0f, 0x0e, 0x0d, 0x0c, 0x0b, 0x0a, 0x09, 0x08,
                   0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x00};
  TestSetValueFromData(etalon256, src, 32, lldb::ByteOrder::eByteOrderBig,
                       RegisterValue::eTypeUIntN);
}

static const Scalar etalon257(APInt(512, 0x0000000000000001ull) << 4 * 64 |
                              APInt(512, 0x1f1e1d1c1b1a1918ull) << 3 * 64 |
                              APInt(512, 0x1716151413121110ull) << 2 * 64 |
                              APInt(512, 0x0f0e0d0c0b0a0908ull) << 1 * 64 |
                              APInt(512, 0x0706050403020100ull) << 0 * 64);

TEST(RegisterValueTest, SetValueFromData_257_le) {
  uint8_t src[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
                   0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11,
                   0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a,
                   0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x01};
  TestSetValueFromData(etalon257, src, 33, lldb::ByteOrder::eByteOrderLittle,
                       RegisterValue::eTypeUIntN);
}

TEST(RegisterValueTest, SetValueFromData_257_be) {
  uint8_t src[] = {0x01, 0x1f, 0x1e, 0x1d, 0x1c, 0x1b, 0x1a, 0x19, 0x18,
                   0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11, 0x10, 0x0f,
                   0x0e, 0x0d, 0x0c, 0x0b, 0x0a, 0x09, 0x08, 0x07, 0x06,
                   0x05, 0x04, 0x03, 0x02, 0x01, 0x00};
  TestSetValueFromData(etalon257, src, 33, lldb::ByteOrder::eByteOrderBig,
                       RegisterValue::eTypeUIntN);
}
