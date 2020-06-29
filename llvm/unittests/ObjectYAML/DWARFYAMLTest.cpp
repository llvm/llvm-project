//===- DWARFYAMLTest.cpp - Tests for DWARFYAML.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ObjectYAML/DWARFYAML.h"
#include "llvm/ObjectYAML/DWARFEmitter.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(DebugAddrSection, TestParseDebugAddrYAML) {
  StringRef Yaml = R"(
debug_addr:
  - Format:  DWARF64
    Length:  0x1234
    Version: 5
)";
  auto SectionsOrErr = DWARFYAML::emitDebugSections(Yaml);
  EXPECT_THAT_EXPECTED(SectionsOrErr, Succeeded());
}

TEST(DebugAddrSection, TestMissingVersion) {
  StringRef Yaml = R"(
debug_addr:
  - Format: DWARF64
    Length: 0x1234
)";
  auto SectionsOrErr = DWARFYAML::emitDebugSections(Yaml);
  EXPECT_THAT_ERROR(SectionsOrErr.takeError(),
                    FailedWithMessage("missing required key 'Version'"));
}

TEST(DebugAddrSection, TestUnexpectedKey) {
  StringRef Yaml = R"(
debug_addr:
  - Format:  DWARF64
    Length:  0x1234
    Version: 5
    Blah:    unexpected
)";
  auto SectionsOrErr = DWARFYAML::emitDebugSections(Yaml);
  EXPECT_THAT_ERROR(SectionsOrErr.takeError(),
                    FailedWithMessage("unknown key 'Blah'"));
}
