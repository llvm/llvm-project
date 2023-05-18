//===- unittest/DebugInfo/CodeView/GUIDFormatTest.cpp - GUID formatting ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/CodeView/Formatters.h"
#include "llvm/DebugInfo/CodeView/GUID.h"
#include "llvm/Support/FormatVariadic.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::codeview;

// Variant 1 UUIDs, nowadays the most common variant, are encoded in a
// big-endian format.
// For example, 00112233-4455-6677-8899-aabbccddeeff is encoded as the bytes
// 00 11 22 33 44 55 66 77 88 99 aa bb cc dd ee ff
//
// Variant 2 UUIDs, historically used in Microsoft's COM/OLE libraries, use a
// mixed-endian format, whereby the first three components of the UUID are
// little-endian, and the last two are big-endian.
// For example, 00112233-4455-6677-8899-aabbccddeeff is encoded as the bytes
// 33 22 11 00 55 44 77 66 88 99 aa bb cc dd ee ff.
//
// Note: Only Variant 2 UUIDs are tested.
namespace {

using GuidPair = std::pair<StringRef, GUID>;
using GuidData = SmallVector<GuidPair>;

void checkData(GuidData &Data) {
  for (auto Item : Data) {
    std::string GuidText(formatv("{0}", Item.second).str());
    StringRef Scalar(GuidText);

    // GUID strings are 38 characters long.
    EXPECT_EQ(Scalar.size(), size_t(38));

    // GUID must be enclosed in {}
    EXPECT_EQ(Scalar.front(), '{');
    EXPECT_EQ(Scalar.back(), '}');

    Scalar = Scalar.substr(1, Scalar.size() - 2);
    SmallVector<StringRef, 6> Component;
    Scalar.split(Component, '-', 5);

    // GUID must have 5 components.
    EXPECT_EQ(Component.size(), size_t(5));

    // GUID components are properly delineated with dashes.
    EXPECT_EQ(Scalar[8], '-');
    EXPECT_EQ(Scalar[13], '-');
    EXPECT_EQ(Scalar[18], '-');
    EXPECT_EQ(Scalar[23], '-');

    // GUID only contains hex digits.
    struct {
      support::ulittle32_t Data0;
      support::ulittle16_t Data1;
      support::ulittle16_t Data2;
      support::ubig16_t Data3;
      support::ubig64_t Data4;
    } G = {};
    EXPECT_TRUE(to_integer(Component[0], G.Data0, 16));
    EXPECT_TRUE(to_integer(Component[1], G.Data1, 16));
    EXPECT_TRUE(to_integer(Component[2], G.Data2, 16));
    EXPECT_TRUE(to_integer(Component[3], G.Data3, 16));
    EXPECT_TRUE(to_integer(Component[4], G.Data4, 16));

    // Check the values are the same.
    EXPECT_EQ(Scalar, Item.first);
  }
}

TEST(GUIDFormatTest, ValidateFormat) {
  // Shifting 2 (0x00)
  GuidData Data = {
      // Non-zero values in all components.
      {"11223344-5566-7788-99AA-BBCCDDEEFFAA",
       {0x44, 0x33, 0x22, 0x11, 0x66, 0x55, 0x88, 0x77, 0x99, 0xaa, 0xbb, 0xcc,
        0xdd, 0xee, 0xff, 0xaa}},

      // Zero values in all components.
      {"00000000-0000-0000-0000-000000000000",
       {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00}},

      // Shift 2 (0x00) across all components
      {"00003344-5566-7788-99AA-BBCCDDEEFFAA",
       {0x44, 0x33, 0x00, 0x00, 0x66, 0x55, 0x88, 0x77, 0x99, 0xaa, 0xbb, 0xcc,
        0xdd, 0xee, 0xff, 0xaa}},
      {"11000044-5566-7788-99AA-BBCCDDEEFFAA",
       {0x44, 0x00, 0x00, 0x11, 0x66, 0x55, 0x88, 0x77, 0x99, 0xaa, 0xbb, 0xcc,
        0xdd, 0xee, 0xff, 0xaa}},
      {"11220000-5566-7788-99AA-BBCCDDEEFFAA",
       {0x00, 0x00, 0x22, 0x11, 0x66, 0x55, 0x88, 0x77, 0x99, 0xaa, 0xbb, 0xcc,
        0xdd, 0xee, 0xff, 0xaa}},
      {"11223300-0066-7788-99AA-BBCCDDEEFFAA",
       {0x00, 0x33, 0x22, 0x11, 0x66, 0x00, 0x88, 0x77, 0x99, 0xaa, 0xbb, 0xcc,
        0xdd, 0xee, 0xff, 0xaa}},
      {"11223344-0000-7788-99AA-BBCCDDEEFFAA",
       {0x44, 0x33, 0x22, 0x11, 0x00, 0x00, 0x88, 0x77, 0x99, 0xaa, 0xbb, 0xcc,
        0xdd, 0xee, 0xff, 0xaa}},
      {"11223344-5500-0088-99AA-BBCCDDEEFFAA",
       {0x44, 0x33, 0x22, 0x11, 0x00, 0x55, 0x88, 0x00, 0x99, 0xaa, 0xbb, 0xcc,
        0xdd, 0xee, 0xff, 0xaa}},
      {"11223344-5566-0000-99AA-BBCCDDEEFFAA",
       {0x44, 0x33, 0x22, 0x11, 0x66, 0x55, 0x00, 0x00, 0x99, 0xaa, 0xbb, 0xcc,
        0xdd, 0xee, 0xff, 0xaa}},
      {"11223344-5566-7700-00AA-BBCCDDEEFFAA",
       {0x44, 0x33, 0x22, 0x11, 0x66, 0x55, 0x00, 0x77, 0x00, 0xaa, 0xbb, 0xcc,
        0xdd, 0xee, 0xff, 0xaa}},
      {"11223344-5566-7788-0000-BBCCDDEEFFAA",
       {0x44, 0x33, 0x22, 0x11, 0x66, 0x55, 0x88, 0x77, 0x00, 0x00, 0xbb, 0xcc,
        0xdd, 0xee, 0xff, 0xaa}},
      {"11223344-5566-7788-9900-00CCDDEEFFAA",
       {0x44, 0x33, 0x22, 0x11, 0x66, 0x55, 0x88, 0x77, 0x99, 0x00, 0x00, 0xcc,
        0xdd, 0xee, 0xff, 0xaa}},
      {"11223344-5566-7788-99AA-0000DDEEFFAA",
       {0x44, 0x33, 0x22, 0x11, 0x66, 0x55, 0x88, 0x77, 0x99, 0xaa, 0x00, 0x00,
        0xdd, 0xee, 0xff, 0xaa}},
      {"11223344-5566-7788-99AA-BB0000EEFFAA",
       {0x44, 0x33, 0x22, 0x11, 0x66, 0x55, 0x88, 0x77, 0x99, 0xaa, 0xbb, 0x00,
        0x00, 0xee, 0xff, 0xaa}},
      {"11223344-5566-7788-99AA-BBCC0000FFAA",
       {0x44, 0x33, 0x22, 0x11, 0x66, 0x55, 0x88, 0x77, 0x99, 0xaa, 0xbb, 0xcc,
        0x00, 0x00, 0xff, 0xaa}},
      {"11223344-5566-7788-99AA-BBCCDD0000AA",
       {0x44, 0x33, 0x22, 0x11, 0x66, 0x55, 0x88, 0x77, 0x99, 0xaa, 0xbb, 0xcc,
        0xdd, 0x00, 0x00, 0xaa}},
      {"11223344-5566-7788-99AA-BBCCDDEE0000",
       {0x44, 0x33, 0x22, 0x11, 0x66, 0x55, 0x88, 0x77, 0x99, 0xaa, 0xbb, 0xcc,
        0xdd, 0xee, 0x00, 0x00}},
  };

  checkData(Data);
}
} // namespace
