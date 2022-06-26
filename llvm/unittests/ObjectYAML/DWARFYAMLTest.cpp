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

template <class T> static Error parseDWARFYAML(StringRef Yaml, T &Data) {
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

  return Error::success();
}

TEST(DebugAddrSection, TestParseDebugAddrYAML) {
  StringRef Yaml = R"(
debug_addr:
  - Format:  DWARF64
    Length:  0x1234
    Version: 5
)";
  DWARFYAML::Data Data;
  EXPECT_THAT_ERROR(parseDWARFYAML(Yaml, Data), Succeeded());
}

TEST(DebugAddrSection, TestMissingVersion) {
  StringRef Yaml = R"(
Format: DWARF64
Length: 0x1234
)";
  DWARFYAML::AddrTableEntry AddrTableEntry;
  EXPECT_THAT_ERROR(parseDWARFYAML(Yaml, AddrTableEntry),
                    FailedWithMessage("missing required key 'Version'"));
}

TEST(DebugAddrSection, TestUnexpectedKey) {
  StringRef Yaml = R"(
Format:  DWARF64
Length:  0x1234
Version: 5
Blah:    unexpected
)";
  DWARFYAML::AddrTableEntry AddrTableEntry;
  EXPECT_THAT_ERROR(parseDWARFYAML(Yaml, AddrTableEntry),
                    FailedWithMessage("unknown key 'Blah'"));
}

TEST(DebugPubSection, TestDebugPubSection) {
  StringRef Yaml = R"(
debug_pubnames:
  Length:        0x1234
  Version:       2
  UnitOffset:    0x4321
  UnitSize:      0x00
  Entries:
    - DieOffset:  0x1234
      Name:       abc
    - DieOffset:  0x4321
      Name:       def
debug_pubtypes:
  Length:        0x1234
  Version:       2
  UnitOffset:    0x4321
  UnitSize:      0x00
  Entries:
    - DieOffset:  0x1234
      Name:       abc
    - DieOffset:  0x4321
      Name:       def
)";
  DWARFYAML::Data Data;
  ASSERT_THAT_ERROR(parseDWARFYAML(Yaml, Data), Succeeded());

  ASSERT_TRUE(Data.PubNames.hasValue());
  DWARFYAML::PubSection PubNames = Data.PubNames.getValue();

  ASSERT_EQ(PubNames.Entries.size(), 2u);
  EXPECT_EQ((uint32_t)PubNames.Entries[0].DieOffset, 0x1234u);
  EXPECT_EQ(PubNames.Entries[0].Name, "abc");
  EXPECT_EQ((uint32_t)PubNames.Entries[1].DieOffset, 0x4321u);
  EXPECT_EQ(PubNames.Entries[1].Name, "def");

  ASSERT_TRUE(Data.PubTypes.hasValue());
  DWARFYAML::PubSection PubTypes = Data.PubTypes.getValue();

  ASSERT_EQ(PubTypes.Entries.size(), 2u);
  EXPECT_EQ((uint32_t)PubTypes.Entries[0].DieOffset, 0x1234u);
  EXPECT_EQ(PubTypes.Entries[0].Name, "abc");
  EXPECT_EQ((uint32_t)PubTypes.Entries[1].DieOffset, 0x4321u);
  EXPECT_EQ(PubTypes.Entries[1].Name, "def");
}

TEST(DebugPubSection, TestUnexpectedDescriptor) {
  StringRef Yaml = R"(
debug_pubnames:
  Length:        0x1234
  Version:       2
  UnitOffset:    0x4321
  UnitSize:      0x00
  Entries:
    - DieOffset:  0x1234
      Descriptor: 0x12
      Name:       abcd
)";
  DWARFYAML::Data Data;
  EXPECT_THAT_ERROR(parseDWARFYAML(Yaml, Data),
                    FailedWithMessage("unknown key 'Descriptor'"));
}

TEST(DebugGNUPubSection, TestDebugGNUPubSections) {
  StringRef Yaml = R"(
debug_gnu_pubnames:
  Length:        0x1234
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
  Length:        0x1234
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
  DWARFYAML::Data Data;
  ASSERT_THAT_ERROR(parseDWARFYAML(Yaml, Data), Succeeded());

  ASSERT_TRUE(Data.GNUPubNames.hasValue());
  DWARFYAML::PubSection GNUPubNames = Data.GNUPubNames.getValue();

  ASSERT_EQ(GNUPubNames.Entries.size(), 2u);
  EXPECT_EQ((uint32_t)GNUPubNames.Entries[0].DieOffset, 0x1234u);
  EXPECT_EQ((uint8_t)GNUPubNames.Entries[0].Descriptor, 0x12);
  EXPECT_EQ(GNUPubNames.Entries[0].Name, "abc");
  EXPECT_EQ((uint32_t)GNUPubNames.Entries[1].DieOffset, 0x4321u);
  EXPECT_EQ((uint8_t)GNUPubNames.Entries[1].Descriptor, 0x34);
  EXPECT_EQ(GNUPubNames.Entries[1].Name, "def");

  ASSERT_TRUE(Data.GNUPubTypes.hasValue());
  DWARFYAML::PubSection GNUPubTypes = Data.GNUPubTypes.getValue();

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
  Length:        0x1234
  Version:       2
  UnitOffset:    0x4321
  UnitSize:      0x00
  Entries:
    - DieOffset: 0x1234
      Name:      abcd
)";
  DWARFYAML::Data Data;
  EXPECT_THAT_ERROR(parseDWARFYAML(Yaml, Data),
                    FailedWithMessage("missing required key 'Descriptor'"));
}
