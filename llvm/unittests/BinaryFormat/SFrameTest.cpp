//===- SFrameTest.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/BinaryFormat/SFrame.h"
#include "gtest/gtest.h"
#include <type_traits>

using namespace llvm;
using namespace llvm::sframe;

namespace {

template <typename EndianT> class SFrameTest : public testing::Test {
protected:
  static constexpr endianness Endian = EndianT::value;

  // Test structure sizes and triviality.
  static_assert(std::is_trivial_v<Preamble<Endian>>);
  static_assert(sizeof(Preamble<Endian>) == 4);

  static_assert(std::is_trivial_v<Header<Endian>>);
  static_assert(sizeof(Header<Endian>) == 28);

  static_assert(std::is_trivial_v<FuncDescEntry<Endian>>);
  static_assert(sizeof(FuncDescEntry<Endian>) == 20);

  static_assert(std::is_trivial_v<FrameRowEntryAddr1<Endian>>);
  static_assert(sizeof(FrameRowEntryAddr1<Endian>) == 2);

  static_assert(std::is_trivial_v<FrameRowEntryAddr2<Endian>>);
  static_assert(sizeof(FrameRowEntryAddr2<Endian>) == 3);

  static_assert(std::is_trivial_v<FrameRowEntryAddr4<Endian>>);
  static_assert(sizeof(FrameRowEntryAddr4<Endian>) == 5);
};

struct NameGenerator {
  template <typename T> static constexpr const char *GetName(int) {
    if constexpr (T::value == endianness::little)
      return "little";
    if constexpr (T::value == endianness::big)
      return "big";
  }
};
using Types =
    testing::Types<std::integral_constant<endianness, endianness::little>,
                   std::integral_constant<endianness, endianness::big>>;
TYPED_TEST_SUITE(SFrameTest, Types, NameGenerator);

TYPED_TEST(SFrameTest, FDEFlags) {
  FuncDescEntry<TestFixture::Endian> FDE = {};
  EXPECT_EQ(FDE.Info.Info, 0u);
  EXPECT_EQ(FDE.Info.getPAuthKey(), 0);
  EXPECT_EQ(FDE.Info.getFDEType(), FDEType::PCInc);
  EXPECT_EQ(FDE.Info.getFREType(), FREType::Addr1);

  FDE.Info.setPAuthKey(1);
  EXPECT_EQ(FDE.Info.Info, 0x20u);
  EXPECT_EQ(FDE.Info.getPAuthKey(), 1);
  EXPECT_EQ(FDE.Info.getFDEType(), FDEType::PCInc);
  EXPECT_EQ(FDE.Info.getFREType(), FREType::Addr1);

  FDE.Info.setFDEType(FDEType::PCMask);
  EXPECT_EQ(FDE.Info.Info, 0x30u);
  EXPECT_EQ(FDE.Info.getPAuthKey(), 1);
  EXPECT_EQ(FDE.Info.getFDEType(), FDEType::PCMask);
  EXPECT_EQ(FDE.Info.getFREType(), FREType::Addr1);

  FDE.Info.setFREType(FREType::Addr4);
  EXPECT_EQ(FDE.Info.Info, 0x32u);
  EXPECT_EQ(FDE.Info.getPAuthKey(), 1);
  EXPECT_EQ(FDE.Info.getFDEType(), FDEType::PCMask);
  EXPECT_EQ(FDE.Info.getFREType(), FREType::Addr4);
}

TYPED_TEST(SFrameTest, FREFlags) {
  FREInfo<TestFixture::Endian> Info = {};
  EXPECT_EQ(Info.Info, 0u);
  EXPECT_FALSE(Info.isReturnAddressSigned());
  EXPECT_EQ(Info.getOffsetSize(), FREOffset::B1);
  EXPECT_EQ(Info.getOffsetCount(), 0u);
  EXPECT_EQ(Info.getBaseRegister(), BaseReg::FP);

  Info.setReturnAddressSigned(true);
  EXPECT_EQ(Info.Info, 0x80u);
  EXPECT_TRUE(Info.isReturnAddressSigned());
  EXPECT_EQ(Info.getOffsetSize(), FREOffset::B1);
  EXPECT_EQ(Info.getOffsetCount(), 0u);
  EXPECT_EQ(Info.getBaseRegister(), BaseReg::FP);

  Info.setOffsetSize(FREOffset::B4);
  EXPECT_EQ(Info.Info, 0xc0u);
  EXPECT_TRUE(Info.isReturnAddressSigned());
  EXPECT_EQ(Info.getOffsetSize(), FREOffset::B4);
  EXPECT_EQ(Info.getOffsetCount(), 0u);
  EXPECT_EQ(Info.getBaseRegister(), BaseReg::FP);

  Info.setOffsetCount(3);
  EXPECT_EQ(Info.Info, 0xc6u);
  EXPECT_TRUE(Info.isReturnAddressSigned());
  EXPECT_EQ(Info.getOffsetSize(), FREOffset::B4);
  EXPECT_EQ(Info.getOffsetCount(), 3u);
  EXPECT_EQ(Info.getBaseRegister(), BaseReg::FP);

  Info.setBaseRegister(BaseReg::SP);
  EXPECT_EQ(Info.Info, 0xc7u);
  EXPECT_TRUE(Info.isReturnAddressSigned());
  EXPECT_EQ(Info.getOffsetSize(), FREOffset::B4);
  EXPECT_EQ(Info.getOffsetCount(), 3u);
  EXPECT_EQ(Info.getBaseRegister(), BaseReg::SP);
}

} // namespace
