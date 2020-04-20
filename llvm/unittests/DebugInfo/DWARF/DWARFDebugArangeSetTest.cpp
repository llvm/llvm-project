//===- llvm/unittest/DebugInfo/DWARFDebugArangeSetTest.cpp-----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFDebugArangeSet.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

template <size_t SecSize>
void ExpectExtractError(const char (&SecDataRaw)[SecSize],
                        const char *ErrorMessage) {
  DWARFDataExtractor Extractor(StringRef(SecDataRaw, SecSize - 1),
                               /* IsLittleEndian = */ true,
                               /* AddressSize = */ 4);
  DWARFDebugArangeSet Set;
  uint64_t Offset = 0;
  Error E = Set.extract(Extractor, &Offset);
  ASSERT_TRUE(E.operator bool());
  EXPECT_STREQ(ErrorMessage, toString(std::move(E)).c_str());
}

TEST(DWARFDebugArangeSet, LengthExceedsSectionSize) {
  static const char DebugArangesSecRaw[] =
      "\x15\x00\x00\x00" // The length exceeds the section boundaries
      "\x02\x00"         // Version
      "\x00\x00\x00\x00" // Debug Info Offset
      "\x04"             // Address Size
      "\x00"             // Segment Selector Size
      "\x00\x00\x00\x00" // Padding
      "\x00\x00\x00\x00" // Termination tuple
      "\x00\x00\x00\x00";
  ExpectExtractError(
      DebugArangesSecRaw,
      "the length of address range table at offset 0x0 exceeds section size");
}

TEST(DWARFDebugArangeSet, LengthExceedsSectionSizeDWARF64) {
  static const char DebugArangesSecRaw[] =
      "\xff\xff\xff\xff"                 // DWARF64 mark
      "\x15\x00\x00\x00\x00\x00\x00\x00" // The length exceeds the section
                                         // boundaries
      "\x02\x00"                         // Version
      "\x00\x00\x00\x00\x00\x00\x00\x00" // Debug Info Offset
      "\x04"                             // Address Size
      "\x00"                             // Segment Selector Size
                                         // No padding
      "\x00\x00\x00\x00"                 // Termination tuple
      "\x00\x00\x00\x00";
  ExpectExtractError(
      DebugArangesSecRaw,
      "the length of address range table at offset 0x0 exceeds section size");
}

TEST(DWARFDebugArangeSet, UnsupportedAddressSize) {
  static const char DebugArangesSecRaw[] =
      "\x0c\x00\x00\x00"  // Length
      "\x02\x00"          // Version
      "\x00\x00\x00\x00"  // Debug Info Offset
      "\x02"              // Address Size (not supported)
      "\x00"              // Segment Selector Size
                          // No padding
      "\x00\x00\x00\x00"; // Termination tuple
  ExpectExtractError(
      DebugArangesSecRaw,
      "address range table at offset 0x0 has unsupported address size: 2 "
      "(4 and 8 supported)");
}

TEST(DWARFDebugArangeSet, UnsupportedSegmentSelectorSize) {
  static const char DebugArangesSecRaw[] =
      "\x14\x00\x00\x00" // Length
      "\x02\x00"         // Version
      "\x00\x00\x00\x00" // Debug Info Offset
      "\x04"             // Address Size
      "\x04"             // Segment Selector Size (not supported)
                         // No padding
      "\x00\x00\x00\x00" // Termination tuple
      "\x00\x00\x00\x00"
      "\x00\x00\x00\x00";
  ExpectExtractError(
      DebugArangesSecRaw,
      "non-zero segment selector size in address range table at offset 0x0 "
      "is not supported");
}

TEST(DWARFDebugArangeSet, NoTerminationEntry) {
  static const char DebugArangesSecRaw[] =
      "\x14\x00\x00\x00" // Length
      "\x02\x00"         // Version
      "\x00\x00\x00\x00" // Debug Info Offset
      "\x04"             // Address Size
      "\x00"             // Segment Selector Size
      "\x00\x00\x00\x00" // Padding
      "\x00\x00\x00\x00" // Entry: Address
      "\x01\x00\x00\x00" //        Length
      ;                  // No termination tuple
  ExpectExtractError(
      DebugArangesSecRaw,
      "address range table at offset 0x0 is not terminated by null entry");
}

TEST(DWARFDebugArangeSet, ReservedUnitLength) {
  // Note: 12 is the minimum length to pass the basic check for the size of
  // the section. 1 will be automatically subtracted in ExpectExtractError().
  static const char DebugArangesSecRaw[12 + 1] =
      "\xf0\xff\xff\xff"; // Reserved unit length value
  ExpectExtractError(DebugArangesSecRaw,
                     "parsing address ranges table at offset 0x0: unsupported "
                     "reserved unit length of value 0xfffffff0");
}

TEST(DWARFDebugArangeSet, SectionTooShort) {
  // Note: 1 will be automatically subtracted in ExpectExtractError().
  static const char DebugArangesSecRaw[11 + 1] = {0};
  ExpectExtractError(DebugArangesSecRaw,
                     "parsing address ranges table at offset 0x0: unexpected "
                     "end of data at offset 0xb while reading [0xb, 0xc)");
}

TEST(DWARFDebugArangeSet, SectionTooShortDWARF64) {
  // Note: 1 will be automatically subtracted in ExpectExtractError().
  static const char DebugArangesSecRaw[23 + 1] =
      "\xff\xff\xff\xff"; // DWARF64 mark
  ExpectExtractError(DebugArangesSecRaw,
                     "parsing address ranges table at offset 0x0: unexpected "
                     "end of data at offset 0x17 while reading [0x17, 0x18)");
}

TEST(DWARFDebugArangeSet, NoSpaceForEntries) {
  static const char DebugArangesSecRaw[] =
      "\x0c\x00\x00\x00" // Length
      "\x02\x00"         // Version
      "\x00\x00\x00\x00" // Debug Info Offset
      "\x04"             // Address Size
      "\x00"             // Segment Selector Size
      "\x00\x00\x00\x00" // Padding
      ;                  // No entries
  ExpectExtractError(
      DebugArangesSecRaw,
      "address range table at offset 0x0 has an insufficient length "
      "to contain any entries");
}

TEST(DWARFDebugArangeSet, UnevenLength) {
  static const char DebugArangesSecRaw[] =
      "\x1b\x00\x00\x00" // Length (not a multiple of tuple size)
      "\x02\x00"         // Version
      "\x00\x00\x00\x00" // Debug Info Offset
      "\x04"             // Address Size
      "\x00"             // Segment Selector Size
      "\x00\x00\x00\x00" // Padding
      "\x00\x00\x00\x00" // Entry: Address
      "\x01\x00\x00\x00" //        Length
      "\x00\x00\x00\x00" // Termination tuple
      "\x00\x00\x00\x00";
  ExpectExtractError(
      DebugArangesSecRaw,
      "address range table at offset 0x0 has length that is not a multiple "
      "of the tuple size");
}

TEST(DWARFDebugArangeSet, ZeroLengthEntry) {
  static const char DebugArangesSecRaw[] =
      "\x24\x00\x00\x00" // Length
      "\x02\x00"         // Version
      "\x00\x00\x00\x00" // Debug Info Offset
      "\x04"             // Address Size
      "\x00"             // Segment Selector Size
      "\x00\x00\x00\x00" // Padding
      "\x00\x00\x00\x00" // Entry1: Address
      "\x01\x00\x00\x00" //         Length
      "\x01\x00\x00\x00" // Entry2: Address
      "\x00\x00\x00\x00" //         Length (invalid)
      "\x00\x00\x00\x00" // Termination tuple
      "\x00\x00\x00\x00";
  ExpectExtractError(
      DebugArangesSecRaw,
      "address range table at offset 0x0 has an invalid tuple (length = 0) "
      "at offset 0x18");
}

} // end anonymous namespace
