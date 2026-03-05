//===- DWARFListTableTest.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFListTable.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(DWARFListTableHeader, TruncatedLength) {
  static const char SecData[] = "\x33\x22\x11"; // Truncated DWARF32 length
  DWARFDataExtractor Extractor(StringRef(SecData, sizeof(SecData) - 1),
                               /*isLittleEndian=*/true,
                               /*AddrSize=*/4);
  DWARFListTableHeader Header(/*SectionName=*/".debug_rnglists",
                              /*ListTypeString=*/"range");
  uint64_t Offset = 0;
  EXPECT_THAT_ERROR(
      Header.extract(Extractor, &Offset),
      FailedWithMessage(
          "parsing .debug_rnglists table at offset 0x0: unexpected end of data "
          "at offset 0x3 while reading [0x0, 0x4)"));
  // length() is expected to return 0 to indicate that the unit length field
  // can not be parsed and so we can not, for example, skip the current set
  // to continue parsing from the next one.
  EXPECT_EQ(Header.length(), 0u);
}

TEST(DWARFListTableHeader, TruncatedLengthDWARF64) {
  static const char SecData[] =
      "\xff\xff\xff\xff"      // DWARF64 mark
      "\x55\x44\x33\x22\x11"; // Truncated DWARF64 length
  DWARFDataExtractor Extractor(StringRef(SecData, sizeof(SecData) - 1),
                               /*isLittleEndian=*/true,
                               /*AddrSize=*/4);
  DWARFListTableHeader Header(/*SectionName=*/".debug_rnglists",
                              /*ListTypeString=*/"range");
  uint64_t Offset = 0;
  EXPECT_THAT_ERROR(
      Header.extract(Extractor, &Offset),
      FailedWithMessage(
          "parsing .debug_rnglists table at offset 0x0: unexpected end of data "
          "at offset 0x9 while reading [0x4, 0xc)"));
  // length() is expected to return 0 to indicate that the unit length field
  // can not be parsed and so we can not, for example, skip the current set
  // to continue parsing from the next one.
  EXPECT_EQ(Header.length(), 0u);
}

TEST(DWARFListTableHeader, TruncatedHeader) {
  static const char SecData[] = "\x02\x00\x00\x00" // Length
                                "\x05\x00";        // Version
  DWARFDataExtractor Extractor(StringRef(SecData, sizeof(SecData) - 1),
                               /*isLittleEndian=*/true,
                               /*AddrSize=*/4);
  DWARFListTableHeader Header(/*SectionName=*/".debug_rnglists",
                              /*ListTypeString=*/"range");
  uint64_t Offset = 0;
  EXPECT_THAT_ERROR(
      Header.extract(Extractor, &Offset),
      FailedWithMessage(".debug_rnglists table at offset 0x0 has too small "
                        "length (0x6) to contain a complete header"));
  // length() is expected to return the full length of the set if the unit
  // length field is read, even if an error occurred during the parsing,
  // to allow skipping the current set and continue parsing from the next one.
  EXPECT_EQ(Header.length(), 6u);
}

TEST(DWARFListTableHeader, OffsetEntryCount) {
  static const char SecData[] = "\x10\x00\x00\x00" // Length
                                "\x05\x00"         // Version
                                "\x08"             // Address size
                                "\x00"             // Segment selector size
                                "\x01\x00\x00\x00" // Offset entry count
                                "\x04\x00\x00\x00" // offset[0]
                                "\x04"             // DW_RLE_offset_pair
                                "\x01"             // ULEB128 starting offset
                                "\x02"             // ULEB128 ending offset
                                "\x00";            // DW_RLE_end_of_list
  DWARFDataExtractor Extractor(StringRef(SecData, sizeof(SecData) - 1),
                               /*isLittleEndian=*/true,
                               /*AddrSize=*/4);
  DWARFListTableHeader Header(/*SectionName=*/".debug_rnglists",
                              /*ListTypeString=*/"range");
  uint64_t Offset = 0;
  EXPECT_FALSE(!!Header.extract(Extractor, &Offset));
  std::optional<uint64_t> Offset0 = Header.getOffsetEntry(Extractor, 0);
  EXPECT_TRUE(!!Offset0);
  EXPECT_EQ(Offset0, uint64_t(4));
  std::optional<uint64_t> Offset1 = Header.getOffsetEntry(Extractor, 1);
  EXPECT_FALSE(!!Offset1);
  EXPECT_EQ(Header.length(), sizeof(SecData) - 1);
}

struct VersionTestCase {
  uint16_t Version;
  bool Supported;
};

struct ListTableVersionFixture
    : public testing::TestWithParam<VersionTestCase> {};

TEST_P(ListTableVersionFixture, VersionTest) {
  // Tests that we correctly reject unsupported DWARF versions.

  const auto [Version, Supported] = GetParam();

  std::string SecData;
  llvm::raw_string_ostream OS(SecData);
  OS.write("\x10\x00\x00\x00", 4); // Length
  llvm::support::endian::write<uint16_t>(OS, Version, llvm::endianness::little);
  OS.write("\x08", 1);             // Address size
  OS.write("\x00", 1);             // Segment selector size
  OS.write("\x01\x00\x00\x00", 4); // Offset entry count
  OS.write("\x04\x00\x00\x00", 4); // offset[0]
  OS.write("\x04", 1);             // DW_RLE_offset_pair
  OS.write("\x01", 1);             // ULEB128 starting offset
  OS.write("\x02", 1);             // ULEB128 ending offset
  OS.write("\x00", 1);             // DW_RLE_end_of_list
  OS.flush();

  DWARFDataExtractor Extractor(SecData,
                               /*isLittleEndian=*/true,
                               /*AddrSize=*/4);
  DWARFListTableHeader Header(/*SectionName=*/".debug_rnglists",
                              /*ListTypeString=*/"range");
  uint64_t Offset = 0;
  auto Err = Header.extract(Extractor, &Offset);

  if (Supported) {
    EXPECT_THAT_ERROR(std::move(Err), llvm::Succeeded());
    EXPECT_EQ(Header.getVersion(), Version);
  } else {
    EXPECT_THAT_ERROR(std::move(Err),
                      llvm::FailedWithMessage(
                          llvm::formatv("unrecognised .debug_rnglists table "
                                        "version {0} in table at offset 0x0",
                                        Version)
                              .str()));
  }
}

VersionTestCase g_version_test_cases[] = {
    {/* 1 less than min. */ 4, false},
    {/* 1 above max. */ 7, false},
    {/* maximum */ 0xFFFF, false},

    // Supported Versions
    {5, true},
    {6, true},
};

INSTANTIATE_TEST_SUITE_P(UnsupportedVersionTestParams, ListTableVersionFixture,
                         testing::ValuesIn(g_version_test_cases));

} // end anonymous namespace
