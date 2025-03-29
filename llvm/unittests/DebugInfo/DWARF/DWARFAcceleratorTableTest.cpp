//===- DWARFAcceleratorTableTest.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFAcceleratorTable.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/ObjectYAML/DWARFEmitter.h"
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

TEST(DWARFDebugNames, BasicTestEntries) {
  const char *Yamldata = R"(
--- !ELF
  debug_str:
    - 'NameType1'
    - 'NameType2'

  debug_names:
    Abbreviations:
    - Code:   0x1
      Tag: DW_TAG_namespace
      Indices:
        - Idx:   DW_IDX_compile_unit
          Form:  DW_FORM_data4
        - Idx:   DW_IDX_die_offset
          Form:  DW_FORM_ref4
    Entries:
    - Name:   0x0  # strp to NameType1
      Code:   0x1
      Values:
        - 0x0      # Compile unit
        - 0x0      # DIE Offset
    - Name:   0xa  # strp to NameType2
      Code:   0x1
      Values:
        - 0x1      # Compile unit
        - 0x1      # DIE Offset
    - Name:   0x0  # strp to NameType1
      Code:   0x1
      Values:
        - 0x2     # Compile unit
        - 0x2     # DIE Offset

)";

  Expected<StringMap<std::unique_ptr<MemoryBuffer>>> Sections =
      DWARFYAML::emitDebugSections(Yamldata,
                                   /*IsLittleEndian=*/true,
                                   /*Is64BitAddrSize=*/true);
  ASSERT_THAT_EXPECTED(Sections, Succeeded());
  auto Ctx = DWARFContext::create(*Sections, 4, /*isLittleEndian=*/true);
  const DWARFDebugNames &DebugNames = Ctx->getDebugNames();
  ASSERT_NE(DebugNames.begin(), DebugNames.end());
  const DWARFDebugNames::NameIndex &NameIndex = *DebugNames.begin();

  ASSERT_EQ(NameIndex.getNameCount(), 2u);
  ASSERT_EQ(NameIndex.getBucketCount(), 0u);
  ASSERT_EQ(NameIndex.getCUCount(), 1u);
  ASSERT_EQ(NameIndex.getCUOffset(0), 0u);
  ASSERT_EQ(NameIndex.getForeignTUCount(), 0u);
  ASSERT_EQ(NameIndex.getLocalTUCount(), 0u);

  // Check "NameEntries": there should be one per unique name.
  // These are indexed starting on 1.
  DWARFDebugNames::NameTableEntry FirstEntry = NameIndex.getNameTableEntry(1);
  ASSERT_EQ(FirstEntry.getString(), StringRef("NameType1"));
  DWARFDebugNames::NameTableEntry SecondEntry = NameIndex.getNameTableEntry(2);
  ASSERT_EQ(SecondEntry.getString(), StringRef("NameType2"));

  SmallVector<DWARFDebugNames::Entry> FirstNameEntries =
      to_vector_of<DWARFDebugNames::Entry>(NameIndex.equal_range("NameType1"));
  ASSERT_EQ(FirstNameEntries.size(), 2u);
  ASSERT_EQ(FirstNameEntries[0].getCUIndex(), 0u);
  ASSERT_EQ(FirstNameEntries[1].getCUIndex(), 0x2);
  ASSERT_EQ(FirstNameEntries[0].getDIEUnitOffset(), 0x0);
  ASSERT_EQ(FirstNameEntries[1].getDIEUnitOffset(), 0x2);

  SmallVector<DWARFDebugNames::Entry> SecondNameEntries =
      to_vector_of<DWARFDebugNames::Entry>(NameIndex.equal_range("NameType2"));
  ASSERT_EQ(SecondNameEntries.size(), 1u);
  ASSERT_EQ(SecondNameEntries[0].getCUIndex(), 0x1);
  ASSERT_EQ(SecondNameEntries[0].getDIEUnitOffset(), 0x1);
}

TEST(DWARFDebugNames, ParentEntries) {
  const char *Yamldata = R"(
--- !ELF
  debug_str:
    - 'Name1'
    - 'Name2'
    - 'Name3'
    - 'Name4'
  debug_names:
    Abbreviations:
    - Code:   0x11
      Tag: DW_TAG_namespace
      Indices:
        - Idx:   DW_IDX_parent
          Form:  DW_FORM_flag_present
        - Idx:   DW_IDX_die_offset
          Form:  DW_FORM_ref4
    - Code:   0x22
      Tag: DW_TAG_namespace
      Indices:
        - Idx:   DW_IDX_parent
          Form:  DW_FORM_ref4
        - Idx:   DW_IDX_die_offset
          Form:  DW_FORM_ref4
    - Code:   0x33
      Tag: DW_TAG_namespace
      Indices:
        - Idx:   DW_IDX_die_offset
          Form:  DW_FORM_ref4
    Entries:
    - Name:   0x0  # strp to Name1
      Code:   0x11
      Values:
        - 0x0      # Die offset
    - Name:   0x6  # strp to Name2
      Code:   0x22
      Values:
        - 0x0      # Parent = First entry
        - 0x1      # Die offset
    - Name:   0xc  # strp to Name3
      Code:   0x22
      Values:
        - 0x6      # Parent = Second entry
        - 0x1      # Die offset
    - Name:   0x12  # strp to Name4
      Code:   0x33
      Values:
        - 0x1      # Die offset
)";

  Expected<StringMap<std::unique_ptr<MemoryBuffer>>> Sections =
      DWARFYAML::emitDebugSections(Yamldata,
                                   /*IsLittleEndian=*/true,
                                   /*Is64BitAddrSize=*/true);
  ASSERT_THAT_EXPECTED(Sections, Succeeded());
  auto Ctx = DWARFContext::create(*Sections, 4, /*isLittleEndian=*/true);
  const DWARFDebugNames &DebugNames = Ctx->getDebugNames();
  ASSERT_NE(DebugNames.begin(), DebugNames.end());
  const DWARFDebugNames::NameIndex &NameIndex = *DebugNames.begin();

  SmallVector<DWARFDebugNames::Entry> Name1Entries =
      to_vector_of<DWARFDebugNames::Entry>(NameIndex.equal_range("Name1"));
  ASSERT_EQ(Name1Entries.size(), 1u);
  Expected<std::optional<DWARFDebugNames::Entry>> Name1Parent =
      Name1Entries[0].getParentDIEEntry();
  ASSERT_THAT_EXPECTED(Name1Parent, Succeeded());
  ASSERT_EQ(*Name1Parent, std::nullopt); // Name1 has no parent

  SmallVector<DWARFDebugNames::Entry> Name2Entries =
      to_vector_of<DWARFDebugNames::Entry>(NameIndex.equal_range("Name2"));
  ASSERT_EQ(Name2Entries.size(), 1u);
  Expected<std::optional<DWARFDebugNames::Entry>> Name2Parent =
      Name2Entries[0].getParentDIEEntry();
  ASSERT_THAT_EXPECTED(Name2Parent, Succeeded());
  ASSERT_EQ((**Name2Parent).getDIEUnitOffset(), 0x0); // Name2 parent == Name1

  SmallVector<DWARFDebugNames::Entry> Name3Entries =
      to_vector_of<DWARFDebugNames::Entry>(NameIndex.equal_range("Name3"));
  ASSERT_EQ(Name3Entries.size(), 1u);
  Expected<std::optional<DWARFDebugNames::Entry>> Name3Parent =
      Name3Entries[0].getParentDIEEntry();
  ASSERT_THAT_EXPECTED(Name3Parent, Succeeded());
  ASSERT_EQ((**Name3Parent).getDIEUnitOffset(), 0x1); // Name3 parent == Name2

  SmallVector<DWARFDebugNames::Entry> Name4Entries =
      to_vector_of<DWARFDebugNames::Entry>(NameIndex.equal_range("Name4"));
  ASSERT_EQ(Name4Entries.size(), 1u);
  ASSERT_FALSE(Name4Entries[0].hasParentInformation());
}

TEST(DWARFDebugNames, InvalidAbbrevCode) {
  const char *Yamldata = R"(
--- !ELF
  debug_str:
    - 'NameType1'

  debug_names:
    Abbreviations:
    - Code:   0x1
      Tag: DW_TAG_namespace
      Indices:
        - Idx:   DW_IDX_compile_unit
          Form:  DW_FORM_data4
    Entries:
    - Name:   0x0  # strp to NameType1
      Code:   0x123456
      Values:
        - 0x0      # Compile unit
)";

  Expected<StringMap<std::unique_ptr<MemoryBuffer>>> Sections =
      DWARFYAML::emitDebugSections(Yamldata,
                                   /*IsLittleEndian=*/true,
                                   /*Is64BitAddrSize=*/true);
  ASSERT_THAT_EXPECTED(
      Sections,
      FailedWithMessage("did not find an Abbreviation for this code"));
}

TEST(DWARFDebugNames, InvalidNumOfValues) {
  const char *Yamldata = R"(
--- !ELF
  debug_str:
    - 'NameType1'

  debug_names:
    Abbreviations:
    - Code:   0x1
      Tag: DW_TAG_namespace
      Indices:
        - Idx:   DW_IDX_compile_unit
          Form:  DW_FORM_data4
    Entries:
    - Name:   0x0  # strp to NameType1
      Code:   0x1
      Values:
        - 0x0      # Compile unit
        - 0x0      # Compile unit
        - 0x0      # Compile unit
)";

  Expected<StringMap<std::unique_ptr<MemoryBuffer>>> Sections =
      DWARFYAML::emitDebugSections(Yamldata,
                                   /*IsLittleEndian=*/true,
                                   /*Is64BitAddrSize=*/true);
  ASSERT_THAT_EXPECTED(
      Sections, FailedWithMessage(
                    "mismatch between provided and required number of values"));
}

TEST(DWARFDebugNames, UnsupportedForm) {
  const char *Yamldata = R"(
--- !ELF
  debug_str:
    - 'NameType1'

  debug_names:
    Abbreviations:
    - Code:   0x1
      Tag: DW_TAG_namespace
      Indices:
        - Idx:   DW_IDX_compile_unit
          Form:  DW_FORM_strx
    Entries:
    - Name:   0x0  # strp to NameType1
      Code:   0x1
      Values:
        - 0x0      # Compile unit
)";

  Expected<StringMap<std::unique_ptr<MemoryBuffer>>> Sections =
      DWARFYAML::emitDebugSections(Yamldata,
                                   /*IsLittleEndian=*/true,
                                   /*Is64BitAddrSize=*/true);
  ASSERT_THAT_EXPECTED(
      Sections,
      FailedWithMessage("unsupported Form for YAML debug_names emitter"));
}
} // end anonymous namespace
