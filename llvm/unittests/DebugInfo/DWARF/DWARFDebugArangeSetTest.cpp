//===- llvm/unittest/DebugInfo/DWARFDebugArangeSetTest.cpp-----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFDebugArangeSet.h"
#include "llvm/DebugInfo/DWARF/DWARFDataExtractor.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

struct WarningHandler {
  ~WarningHandler() { EXPECT_THAT_ERROR(std::move(Err), Succeeded()); }

  void operator()(Error E) { Err = joinErrors(std::move(Err), std::move(E)); }

  Error getWarning() { return std::move(Err); }

  Error Err = Error::success();
};

template <size_t SecSize>
void ExpectExtractError(const char (&SecDataRaw)[SecSize],
                        const char *ErrorMessage) {
  DWARFDataExtractor Extractor(StringRef(SecDataRaw, SecSize - 1),
                               /* IsLittleEndian = */ true,
                               /* AddressSize = */ 4);
  DWARFDebugArangeSet Set;
  uint64_t Offset = 0;
  WarningHandler Warnings;
  Error E = Set.extract(Extractor, &Offset, Warnings);
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
      "\x03"              // Address Size (not supported)
      "\x00"              // Segment Selector Size
                          // No padding
      "\x00\x00\x00\x00"; // Termination tuple
  ExpectExtractError(
      DebugArangesSecRaw,
      "address range table at offset 0x0 has unsupported address size: 3 "
      "(supported are 2, 4, 8)");
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

TEST(DWARFDebugArangeSet, ZeroAddressEntry) {
  static const char DebugArangesSecRaw[] =
      "\x1c\x00\x00\x00" // Length
      "\x02\x00"         // Version
      "\x00\x00\x00\x00" // Debug Info Offset
      "\x04"             // Address Size
      "\x00"             // Segment Selector Size
      "\x00\x00\x00\x00" // Padding
      "\x00\x00\x00\x00" // Entry1: Address
      "\x01\x00\x00\x00" //         Length
      "\x00\x00\x00\x00" // Termination tuple
      "\x00\x00\x00\x00";
  DWARFDataExtractor Extractor(
      StringRef(DebugArangesSecRaw, sizeof(DebugArangesSecRaw) - 1),
      /*IsLittleEndian=*/true,
      /*AddressSize=*/4);
  DWARFDebugArangeSet Set;
  uint64_t Offset = 0;
  ASSERT_THAT_ERROR(Set.extract(Extractor, &Offset, WarningHandler()),
                    Succeeded());
  auto Range = Set.descriptors();
  auto Iter = Range.begin();
  ASSERT_EQ(std::distance(Iter, Range.end()), 1);
  EXPECT_EQ(Iter->Address, 0u);
  EXPECT_EQ(Iter->Length, 1u);
}

TEST(DWARFDebugArangeSet, ZeroLengthEntry) {
  static const char DebugArangesSecRaw[] =
      "\x1c\x00\x00\x00" // Length
      "\x02\x00"         // Version
      "\x00\x00\x00\x00" // Debug Info Offset
      "\x04"             // Address Size
      "\x00"             // Segment Selector Size
      "\x00\x00\x00\x00" // Padding
      "\x01\x00\x00\x00" // Entry1: Address
      "\x00\x00\x00\x00" //         Length
      "\x00\x00\x00\x00" // Termination tuple
      "\x00\x00\x00\x00";
  DWARFDataExtractor Extractor(
      StringRef(DebugArangesSecRaw, sizeof(DebugArangesSecRaw) - 1),
      /*IsLittleEndian=*/true,
      /*AddressSize=*/4);
  DWARFDebugArangeSet Set;
  uint64_t Offset = 0;
  ASSERT_THAT_ERROR(Set.extract(Extractor, &Offset, WarningHandler()),
                    Succeeded());
  auto Range = Set.descriptors();
  auto Iter = Range.begin();
  ASSERT_EQ(std::distance(Iter, Range.end()), 1);
  EXPECT_EQ(Iter->Address, 1u);
  EXPECT_EQ(Iter->Length, 0u);
}

TEST(DWARFDebugArangesSet, PrematureTerminator) {
  static const char DebugArangesSecRaw[] =
      "\x24\x00\x00\x00" // Length
      "\x02\x00"         // Version
      "\x00\x00\x00\x00" // Debug Info Offset
      "\x04"             // Address Size
      "\x00"             // Segment Selector Size
      "\x00\x00\x00\x00" // Padding
      "\x00\x00\x00\x00" // Entry1: Premature
      "\x00\x00\x00\x00" //         terminator
      "\x01\x00\x00\x00" // Entry2: Address
      "\x01\x00\x00\x00" //         Length
      "\x00\x00\x00\x00" // Termination tuple
      "\x00\x00\x00\x00";
  DWARFDataExtractor Extractor(
      StringRef(DebugArangesSecRaw, sizeof(DebugArangesSecRaw) - 1),
      /*IsLittleEndian=*/true,
      /*AddressSize=*/4);
  DWARFDebugArangeSet Set;
  uint64_t Offset = 0;
  WarningHandler Warnings;
  ASSERT_THAT_ERROR(Set.extract(Extractor, &Offset, Warnings), Succeeded());
  auto Range = Set.descriptors();
  auto Iter = Range.begin();
  ASSERT_EQ(std::distance(Iter, Range.end()), 2);
  EXPECT_EQ(Iter->Address, 0u);
  EXPECT_EQ(Iter->Length, 0u);
  ++Iter;
  EXPECT_EQ(Iter->Address, 1u);
  EXPECT_EQ(Iter->Length, 1u);
  EXPECT_THAT_ERROR(
      Warnings.getWarning(),
      FailedWithMessage("address range table at offset 0x0 has a premature "
                        "terminator entry at offset 0x10"));
}

} // end anonymous namespace
