//===- DWARFAcceleratorTableTest.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFAcceleratorTable.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;

static Error ExtractDebugNames(StringRef NamesSecData, StringRef StrSecData) {
  DWARFDataExtractor NamesExtractor(NamesSecData,
                                    /*isLittleEndian=*/true,
                                    /*AddrSize=*/4);
  DataExtractor StrExtractor(StrSecData,
                             /*isLittleEndian=*/true,
                             /*AddrSize=*/4);
  DWARFDebugNames Table(NamesExtractor, StrExtractor);
  return Table.extract();
}

namespace {

TEST(DWARFDebugNames, ReservedUnitLength) {
  static const char NamesSecData[64] =
      "\xf0\xff\xff\xff"; // Reserved unit length value
  EXPECT_THAT_ERROR(
      ExtractDebugNames(StringRef(NamesSecData, sizeof(NamesSecData)),
                        StringRef()),
      FailedWithMessage("parsing .debug_names header at 0x0: unsupported "
                        "reserved unit length of value 0xfffffff0"));
}

TEST(DWARFDebugNames, TooSmallForDWARF64) {
  // DWARF64 header takes at least 44 bytes.
  static const char NamesSecData[43] = "\xff\xff\xff\xff"; // DWARF64 mark
  EXPECT_THAT_ERROR(
      ExtractDebugNames(StringRef(NamesSecData, sizeof(NamesSecData)),
                        StringRef()),
      FailedWithMessage("parsing .debug_names header at 0x0: unexpected end of "
                        "data at offset 0x2b while reading [0x28, 0x2c)"));
}

} // end anonymous namespace
