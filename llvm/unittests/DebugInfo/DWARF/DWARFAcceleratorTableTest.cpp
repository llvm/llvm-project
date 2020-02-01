//===- DWARFAcceleratorTableTest.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFAcceleratorTable.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

void ExpectDebugNamesExtractError(StringRef NamesSecData, StringRef StrSecData,
                                  const char *ErrorMessage) {
  DWARFSection NamesDWARFSection;
  NamesDWARFSection.Data = NamesSecData;
  StringMap<std::unique_ptr<MemoryBuffer>> Sections;
  auto Context = DWARFContext::create(Sections, /* AddrSize = */ 4,
                                      /* isLittleEndian = */ true);
  DWARFDataExtractor NamesExtractor(Context->getDWARFObj(), NamesDWARFSection,
                                    /* isLittleEndian = */ true,
                                    /* AddrSize = */ 4);
  DataExtractor StrExtractor(StrSecData,
                             /* isLittleEndian = */ true,
                             /* AddrSize = */ 4);
  DWARFDebugNames Table(NamesExtractor, StrExtractor);
  Error E = Table.extract();
  ASSERT_TRUE(E.operator bool());
  EXPECT_STREQ(ErrorMessage, toString(std::move(E)).c_str());
}

TEST(DWARFDebugNames, ReservedUnitLength) {
  static const char NamesSecData[64] =
      "\xf0\xff\xff\xff"; // Reserved unit length value
  ExpectDebugNamesExtractError(StringRef(NamesSecData, sizeof(NamesSecData)),
                               StringRef(),
                               "Unsupported reserved unit length value");
}

TEST(DWARFDebugNames, TooSmallForDWARF64) {
  // DWARF64 header takes at least 44 bytes.
  static const char NamesSecData[43] = "\xff\xff\xff\xff"; // DWARF64 mark
  ExpectDebugNamesExtractError(
      StringRef(NamesSecData, sizeof(NamesSecData)), StringRef(),
      "Section too small: cannot read header.");
}

} // end anonymous namespace
