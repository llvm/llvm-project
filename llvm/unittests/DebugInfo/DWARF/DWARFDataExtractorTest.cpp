//===- DWARFDataExtractorTest.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFDataExtractor.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(DWARFDataExtractorTest, getInitialLength) {
  auto GetWithError = [](ArrayRef<uint8_t> Bytes)
      -> Expected<std::tuple<uint64_t, dwarf::DwarfFormat, uint64_t>> {
    DWARFDataExtractor Data(Bytes, /*IsLittleEndian=*/false, /*AddressSize=*/8);
    DWARFDataExtractor::Cursor C(0);
    uint64_t Length;
    dwarf::DwarfFormat Format;
    std::tie(Length, Format) = Data.getInitialLength(C);
    if (C)
      return std::make_tuple(Length, Format, C.tell());

    EXPECT_EQ(Length, 0u);
    EXPECT_EQ(Format, dwarf::DWARF32);
    EXPECT_EQ(C.tell(), 0u);
    return C.takeError();
  };
  auto GetWithoutError = [](ArrayRef<uint8_t> Bytes) {
    DWARFDataExtractor Data(Bytes, /*IsLittleEndian=*/false, /*AddressSize=*/8);
    uint64_t Offset = 0;
    uint64_t Length;
    dwarf::DwarfFormat Format;
    std::tie(Length, Format) = Data.getInitialLength(&Offset);
    return std::make_tuple(Length, Format, Offset);
  };
  auto ErrorResult = std::make_tuple(0, dwarf::DWARF32, 0);

  // Empty data.
  EXPECT_THAT_EXPECTED(GetWithError({}),
                       FailedWithMessage("unexpected end of data"));
  EXPECT_EQ(GetWithoutError({}), ErrorResult);

  // Not long enough for the U32 field.
  EXPECT_THAT_EXPECTED(GetWithError({0x00, 0x01, 0x02}),
                       FailedWithMessage("unexpected end of data"));
  EXPECT_EQ(GetWithoutError({0x00, 0x01, 0x02}), ErrorResult);

  EXPECT_THAT_EXPECTED(
      GetWithError({0x00, 0x01, 0x02, 0x03}),
      HasValue(std::make_tuple(0x00010203, dwarf::DWARF32, 4)));
  EXPECT_EQ(GetWithoutError({0x00, 0x01, 0x02, 0x03}),
            std::make_tuple(0x00010203, dwarf::DWARF32, 4));

  // Zeroes are not an error, but without the Error object it is hard to tell
  // them apart from a failed read.
  EXPECT_THAT_EXPECTED(
      GetWithError({0x00, 0x00, 0x00, 0x00}),
      HasValue(std::make_tuple(0x00000000, dwarf::DWARF32, 4)));
  EXPECT_EQ(GetWithoutError({0x00, 0x00, 0x00, 0x00}),
            std::make_tuple(0x00000000, dwarf::DWARF32, 4));

  // Smallest invalid value.
  EXPECT_THAT_EXPECTED(
      GetWithError({0xff, 0xff, 0xff, 0xf0}),
      FailedWithMessage(
          "unsupported reserved unit length of value 0xfffffff0"));
  EXPECT_EQ(GetWithoutError({0xff, 0xff, 0xff, 0xf0}), ErrorResult);

  // DWARF64 marker without the subsequent length field.
  EXPECT_THAT_EXPECTED(GetWithError({0xff, 0xff, 0xff, 0xff}),
                       FailedWithMessage("unexpected end of data"));
  EXPECT_EQ(GetWithoutError({0xff, 0xff, 0xff, 0xff}), ErrorResult);

  // Not enough data for the U64 length.
  EXPECT_THAT_EXPECTED(
      GetWithError({0xff, 0xff, 0xff, 0xff, 0x00, 0x01, 0x02, 0x03}),
      FailedWithMessage("unexpected end of data"));
  EXPECT_EQ(GetWithoutError({0xff, 0xff, 0xff, 0xff, 0x00, 0x01, 0x02, 0x03}),
            ErrorResult);

  EXPECT_THAT_EXPECTED(
      GetWithError({0xff, 0xff, 0xff, 0xff, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05,
                    0x06, 0x07}),
      HasValue(std::make_tuple(0x0001020304050607, dwarf::DWARF64, 12)));
  EXPECT_EQ(GetWithoutError({0xff, 0xff, 0xff, 0xff, 0x00, 0x01, 0x02, 0x03,
                             0x04, 0x05, 0x06, 0x07}),
            std::make_tuple(0x0001020304050607, dwarf::DWARF64, 12));
}

} // namespace
