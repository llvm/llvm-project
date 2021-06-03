//===-- RegisterValueTest.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/RegisterValue.h"
#include "gtest/gtest.h"

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
  const auto &Get = [](const RV &V) -> llvm::Optional<Scalar> {
    Scalar S;
    if (V.GetScalarValue(S))
      return S;
    return llvm::None;
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
