//===- DWARFYAMLTest.cpp - Tests for DWARFYAML.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ObjectYAML/DWARFYAML.h"
#include "llvm/ObjectYAML/DWARFEmitter.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;

static Expected<DWARFYAML::Data> parseDWARFYAML(StringRef Yaml,
                                                bool IsLittleEndian = false,
                                                bool Is64bit = true) {
  DWARFYAML::Data Data;
  Data.IsLittleEndian = IsLittleEndian;
  Data.Is64bit = Is64bit;

  SMDiagnostic GenerateDiag;
  yaml::Input YIn(
      Yaml, /*Ctxt=*/nullptr,
      [](const SMDiagnostic &Diag, void *DiagContext) {
        *static_cast<SMDiagnostic *>(DiagContext) = Diag;
      },
      &GenerateDiag);

  YIn >> Data;
  if (YIn.error())
    return createStringError(YIn.error(), GenerateDiag.getMessage());

  return Data;
}

TEST(DebugAddrSection, TestParseDebugAddrYAML) {
  StringRef Yaml = R"(
debug_addr:
  - Format:  DWARF64
    Length:  0x1234
    Version: 5
)";
  auto DWARFOrErr = parseDWARFYAML(Yaml);
  EXPECT_THAT_EXPECTED(DWARFOrErr, Succeeded());
}

TEST(DebugAddrSection, TestMissingVersion) {
  StringRef Yaml = R"(
debug_addr:
  - Format: DWARF64
    Length: 0x1234
)";
  auto DWARFOrErr = parseDWARFYAML(Yaml);
  EXPECT_THAT_ERROR(DWARFOrErr.takeError(),
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
  auto DWARFOrErr = parseDWARFYAML(Yaml);
  EXPECT_THAT_ERROR(DWARFOrErr.takeError(),
                    FailedWithMessage("unknown key 'Blah'"));
}

TEST(DebugPubSection, TestDebugPubSection) {
  StringRef Yaml = R"(
debug_pubnames:
  Length:
    TotalLength: 0x1234
  Version:       2
  UnitOffset:    0x4321
  UnitSize:      0x00
  Entries:
    - DieOffset:  0x1234
      Name:       abc
    - DieOffset:  0x4321
      Name:       def
debug_pubtypes:
  Length:
    TotalLength: 0x1234
  Version:       2
  UnitOffset:    0x4321
  UnitSize:      0x00
  Entries:
    - DieOffset:  0x1234
      Name:       abc
    - DieOffset:  0x4321
      Name:       def
)";
  auto DWARFOrErr = parseDWARFYAML(Yaml);
  ASSERT_THAT_EXPECTED(DWARFOrErr, Succeeded());

  ASSERT_TRUE(DWARFOrErr->PubNames.hasValue());
  DWARFYAML::PubSection PubNames = DWARFOrErr->PubNames.getValue();

  ASSERT_EQ(PubNames.Entries.size(), 2u);
  EXPECT_EQ((uint32_t)PubNames.Entries[0].DieOffset, 0x1234u);
  EXPECT_EQ(PubNames.Entries[0].Name, "abc");
  EXPECT_EQ((uint32_t)PubNames.Entries[1].DieOffset, 0x4321u);
  EXPECT_EQ(PubNames.Entries[1].Name, "def");

  ASSERT_TRUE(DWARFOrErr->PubTypes.hasValue());
  DWARFYAML::PubSection PubTypes = DWARFOrErr->PubTypes.getValue();

  ASSERT_EQ(PubTypes.Entries.size(), 2u);
  EXPECT_EQ((uint32_t)PubTypes.Entries[0].DieOffset, 0x1234u);
  EXPECT_EQ(PubTypes.Entries[0].Name, "abc");
  EXPECT_EQ((uint32_t)PubTypes.Entries[1].DieOffset, 0x4321u);
  EXPECT_EQ(PubTypes.Entries[1].Name, "def");
}

TEST(DebugPubSection, TestUnexpectedDescriptor) {
  StringRef Yaml = R"(
debug_pubnames:
  Length:
    TotalLength: 0x1234
  Version:       2
  UnitOffset:    0x4321
  UnitSize:      0x00
  Entries:
    - DieOffset:  0x1234
      Descriptor: 0x12
      Name:       abcd
)";
  auto DWARFOrErr = parseDWARFYAML(Yaml);
  EXPECT_THAT_ERROR(DWARFOrErr.takeError(),
                    FailedWithMessage("unknown key 'Descriptor'"));
}

TEST(DebugGNUPubSection, TestDebugGNUPubSections) {
  StringRef Yaml = R"(
debug_gnu_pubnames:
  Length:
    TotalLength: 0x1234
  Version:       2
  UnitOffset:    0x4321
  UnitSize:      0x00
  Entries:
    - DieOffset:  0x1234
      Descriptor: 0x12
      Name:       abc
    - DieOffset:  0x4321
      Descriptor: 0x34
      Name:       def
debug_gnu_pubtypes:
  Length:
    TotalLength: 0x1234
  Version:       2
  UnitOffset:    0x4321
  UnitSize:      0x00
  Entries:
    - DieOffset:  0x1234
      Descriptor: 0x12
      Name:       abc
    - DieOffset:  0x4321
      Descriptor: 0x34
      Name:       def
)";
  auto DWARFOrErr = parseDWARFYAML(Yaml);
  ASSERT_THAT_EXPECTED(DWARFOrErr, Succeeded());

  ASSERT_TRUE(DWARFOrErr->GNUPubNames.hasValue());
  DWARFYAML::PubSection GNUPubNames = DWARFOrErr->GNUPubNames.getValue();

  ASSERT_EQ(GNUPubNames.Entries.size(), 2u);
  EXPECT_EQ((uint32_t)GNUPubNames.Entries[0].DieOffset, 0x1234u);
  EXPECT_EQ((uint8_t)GNUPubNames.Entries[0].Descriptor, 0x12);
  EXPECT_EQ(GNUPubNames.Entries[0].Name, "abc");
  EXPECT_EQ((uint32_t)GNUPubNames.Entries[1].DieOffset, 0x4321u);
  EXPECT_EQ((uint8_t)GNUPubNames.Entries[1].Descriptor, 0x34);
  EXPECT_EQ(GNUPubNames.Entries[1].Name, "def");

  ASSERT_TRUE(DWARFOrErr->GNUPubTypes.hasValue());
  DWARFYAML::PubSection GNUPubTypes = DWARFOrErr->GNUPubTypes.getValue();

  ASSERT_EQ(GNUPubTypes.Entries.size(), 2u);
  EXPECT_EQ((uint32_t)GNUPubTypes.Entries[0].DieOffset, 0x1234u);
  EXPECT_EQ((uint8_t)GNUPubTypes.Entries[0].Descriptor, 0x12);
  EXPECT_EQ(GNUPubTypes.Entries[0].Name, "abc");
  EXPECT_EQ((uint32_t)GNUPubTypes.Entries[1].DieOffset, 0x4321u);
  EXPECT_EQ((uint8_t)GNUPubTypes.Entries[1].Descriptor, 0x34);
  EXPECT_EQ(GNUPubTypes.Entries[1].Name, "def");
}

TEST(DebugGNUPubSection, TestMissingDescriptor) {
  StringRef Yaml = R"(
debug_gnu_pubnames:
  Length:
    TotalLength: 0x1234
  Version:       2
  UnitOffset:    0x4321
  UnitSize:      0x00
  Entries:
    - DieOffset: 0x1234
      Name:      abcd
)";
  auto DWARFOrErr = parseDWARFYAML(Yaml);
  EXPECT_THAT_ERROR(DWARFOrErr.takeError(),
                    FailedWithMessage("missing required key 'Descriptor'"));
}
