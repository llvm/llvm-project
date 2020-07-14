//===- DWARFListTableTest.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFListTable.h"
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

} // end anonymous namespace
