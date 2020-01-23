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
  DataExtractor Extractor(StringRef(SecDataRaw, SecSize - 1),
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

} // end anonymous namespace
