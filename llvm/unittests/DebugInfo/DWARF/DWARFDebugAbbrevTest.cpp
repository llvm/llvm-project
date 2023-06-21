//===- llvm/unittest/DebugInfo/DWARFDebugAbbrevTest.cpp -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFDebugAbbrev.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DWARF/DWARFAbbreviationDeclaration.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace dwarf;

enum OrderKind : bool { InOrder, OutOfOrder };

void writeValidAbbreviationDeclarations(raw_ostream &OS, uint32_t FirstCode,
                                        OrderKind Order) {
  encodeULEB128(FirstCode, OS);
  encodeULEB128(DW_TAG_compile_unit, OS);
  OS << static_cast<uint8_t>(DW_CHILDREN_yes);
  encodeULEB128(DW_AT_name, OS);
  encodeULEB128(DW_FORM_strp, OS);
  encodeULEB128(0, OS);
  encodeULEB128(0, OS);

  uint32_t SecondCode =
      Order == OrderKind::InOrder ? FirstCode + 1 : FirstCode - 1;

  encodeULEB128(SecondCode, OS);
  encodeULEB128(DW_TAG_subprogram, OS);
  OS << static_cast<uint8_t>(DW_CHILDREN_no);
  encodeULEB128(DW_AT_name, OS);
  encodeULEB128(DW_FORM_strp, OS);
  encodeULEB128(0, OS);
  encodeULEB128(0, OS);
}

void writeMalformedULEB128Value(raw_ostream &OS) {
  OS << static_cast<uint8_t>(0x80);
}

void writeULEB128LargerThan64Bits(raw_ostream &OS) {
  static constexpr llvm::StringRef LargeULEB128 =
      "\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x01";
  OS << LargeULEB128;
}

TEST(DWARFDebugAbbrevTest, DWARFAbbrevDeclSetExtractSuccess) {
  SmallString<64> RawData;
  raw_svector_ostream OS(RawData);
  uint32_t FirstCode = 5;

  writeValidAbbreviationDeclarations(OS, FirstCode, InOrder);
  encodeULEB128(0, OS);

  uint64_t Offset = 0;
  DataExtractor Data(RawData, sys::IsLittleEndianHost, sizeof(uint64_t));
  DWARFAbbreviationDeclarationSet AbbrevSet;
  ASSERT_THAT_ERROR(AbbrevSet.extract(Data, &Offset), Succeeded());
  // The Abbreviation Declarations are in order and contiguous, so we want to
  // make sure that FirstAbbrCode was correctly set.
  EXPECT_EQ(AbbrevSet.getFirstAbbrCode(), FirstCode);

  const DWARFAbbreviationDeclaration *Abbrev5 =
      AbbrevSet.getAbbreviationDeclaration(FirstCode);
  ASSERT_TRUE(Abbrev5);
  EXPECT_EQ(Abbrev5->getTag(), DW_TAG_compile_unit);
  EXPECT_TRUE(Abbrev5->hasChildren());
  EXPECT_EQ(Abbrev5->getNumAttributes(), 1u);

  const DWARFAbbreviationDeclaration *Abbrev6 =
      AbbrevSet.getAbbreviationDeclaration(FirstCode + 1);
  ASSERT_TRUE(Abbrev6);
  EXPECT_EQ(Abbrev6->getTag(), DW_TAG_subprogram);
  EXPECT_FALSE(Abbrev6->hasChildren());
  EXPECT_EQ(Abbrev6->getNumAttributes(), 1u);
}

TEST(DWARFDebugAbbrevTest, DWARFAbbrevDeclSetExtractSuccessOutOfOrder) {
  SmallString<64> RawData;
  raw_svector_ostream OS(RawData);
  uint32_t FirstCode = 2;

  writeValidAbbreviationDeclarations(OS, FirstCode, OutOfOrder);
  encodeULEB128(0, OS);

  uint64_t Offset = 0;
  DataExtractor Data(RawData, sys::IsLittleEndianHost, sizeof(uint64_t));
  DWARFAbbreviationDeclarationSet AbbrevSet;
  ASSERT_THAT_ERROR(AbbrevSet.extract(Data, &Offset), Succeeded());
  // The declarations are out of order, ensure that FirstAbbrCode is UINT32_MAX.
  EXPECT_EQ(AbbrevSet.getFirstAbbrCode(), UINT32_MAX);

  const DWARFAbbreviationDeclaration *Abbrev2 =
      AbbrevSet.getAbbreviationDeclaration(FirstCode);
  ASSERT_TRUE(Abbrev2);
  EXPECT_EQ(Abbrev2->getTag(), DW_TAG_compile_unit);
  EXPECT_TRUE(Abbrev2->hasChildren());
  EXPECT_EQ(Abbrev2->getNumAttributes(), 1u);

  const DWARFAbbreviationDeclaration *Abbrev1 =
      AbbrevSet.getAbbreviationDeclaration(FirstCode - 1);
  ASSERT_TRUE(Abbrev1);
  EXPECT_EQ(Abbrev1->getTag(), DW_TAG_subprogram);
  EXPECT_FALSE(Abbrev1->hasChildren());
  EXPECT_EQ(Abbrev1->getNumAttributes(), 1u);
}

TEST(DWARFDebugAbbrevTest, DWARFAbbreviationDeclSetCodeExtractionError) {
  SmallString<64> RawData;

  // Check for malformed ULEB128.
  {
    raw_svector_ostream OS(RawData);
    writeMalformedULEB128Value(OS);

    uint64_t Offset = 0;
    DataExtractor Data(RawData, sys::IsLittleEndianHost, sizeof(uint64_t));
    DWARFAbbreviationDeclarationSet AbbrevSet;
    ASSERT_THAT_ERROR(
        AbbrevSet.extract(Data, &Offset),
        FailedWithMessage("unable to decode LEB128 at offset 0x00000000: "
                          "malformed uleb128, extends past end"));
    EXPECT_EQ(Offset, 0u);
  }

  RawData.clear();
  // Check for ULEB128 too large to fit into a uin64_t.
  {
    raw_svector_ostream OS(RawData);
    writeULEB128LargerThan64Bits(OS);

    uint64_t Offset = 0;
    DataExtractor Data(RawData, sys::IsLittleEndianHost, sizeof(uint64_t));
    DWARFAbbreviationDeclarationSet AbbrevSet;
    ASSERT_THAT_ERROR(
        AbbrevSet.extract(Data, &Offset),
        FailedWithMessage("unable to decode LEB128 at offset 0x00000000: "
                          "uleb128 too big for uint64"));
    EXPECT_EQ(Offset, 0u);
  }
}

TEST(DWARFDebugAbbrevTest, DWARFAbbreviationDeclSetTagExtractionError) {
  SmallString<64> RawData;
  const uint32_t Code = 1;

  // Check for malformed ULEB128.
  {
    raw_svector_ostream OS(RawData);
    encodeULEB128(Code, OS);
    writeMalformedULEB128Value(OS);

    uint64_t Offset = 0;
    DataExtractor Data(RawData, sys::IsLittleEndianHost, sizeof(uint64_t));
    DWARFAbbreviationDeclarationSet AbbrevSet;
    ASSERT_THAT_ERROR(
        AbbrevSet.extract(Data, &Offset),
        FailedWithMessage("unable to decode LEB128 at offset 0x00000001: "
                          "malformed uleb128, extends past end"));
    // Only the code was extracted correctly.
    EXPECT_EQ(Offset, 1u);
  }

  RawData.clear();
  // Check for ULEB128 too large to fit into a uint64_t.
  {
    raw_svector_ostream OS(RawData);
    encodeULEB128(Code, OS);
    writeULEB128LargerThan64Bits(OS);

    uint64_t Offset = 0;
    DataExtractor Data(RawData, sys::IsLittleEndianHost, sizeof(uint64_t));
    DWARFAbbreviationDeclarationSet AbbrevSet;
    ASSERT_THAT_ERROR(
        AbbrevSet.extract(Data, &Offset),
        FailedWithMessage("unable to decode LEB128 at offset 0x00000001: "
                          "uleb128 too big for uint64"));
    // Only the code was extracted correctly.
    EXPECT_EQ(Offset, 1u);
  }
}

TEST(DWARFDebugAbbrevTest, DWARFAbbreviationDeclSetChildExtractionError) {
  SmallString<64> RawData;
  const uint32_t Code = 1;
  const dwarf::Tag Tag = DW_TAG_compile_unit;

  // We want to make sure that we fail if we reach the end of the stream before
  // reading the 'children' byte.
  raw_svector_ostream OS(RawData);
  encodeULEB128(Code, OS);
  encodeULEB128(Tag, OS);
  uint64_t Offset = 0;
  DataExtractor Data(RawData, sys::IsLittleEndianHost, sizeof(uint64_t));
  DWARFAbbreviationDeclarationSet AbbrevSet;
  ASSERT_THAT_ERROR(
      AbbrevSet.extract(Data, &Offset),
      FailedWithMessage(
          "unexpected end of data at offset 0x2 while reading [0x2, 0x3)"));
  // The code and the tag were extracted correctly.
  EXPECT_EQ(Offset, 2u);
}

TEST(DWARFDebugAbbrevTest, DWARFAbbreviationDeclSetAttributeExtractionError) {
  SmallString<64> RawData;
  const uint32_t Code = 1;
  const dwarf::Tag Tag = DW_TAG_compile_unit;
  const uint8_t Children = DW_CHILDREN_yes;

  // Check for malformed ULEB128.
  {
    raw_svector_ostream OS(RawData);
    encodeULEB128(Code, OS);
    encodeULEB128(Tag, OS);
    OS << Children;
    writeMalformedULEB128Value(OS);

    uint64_t Offset = 0;
    DataExtractor Data(RawData, sys::IsLittleEndianHost, sizeof(uint64_t));
    DWARFAbbreviationDeclarationSet AbbrevSet;
    ASSERT_THAT_ERROR(
        AbbrevSet.extract(Data, &Offset),
        FailedWithMessage("unable to decode LEB128 at offset 0x00000003: "
                          "malformed uleb128, extends past end"));
    // The code, tag, and child byte were extracted correctly.
    EXPECT_EQ(Offset, 3u);
  }

  RawData.clear();
  // Check for ULEB128 too large to fit into a uint64_t.
  {
    raw_svector_ostream OS(RawData);
    encodeULEB128(Code, OS);
    encodeULEB128(Tag, OS);
    OS << Children;
    writeULEB128LargerThan64Bits(OS);

    uint64_t Offset = 0;
    DataExtractor Data(RawData, sys::IsLittleEndianHost, sizeof(uint64_t));
    DWARFAbbreviationDeclarationSet AbbrevSet;
    ASSERT_THAT_ERROR(
        AbbrevSet.extract(Data, &Offset),
        FailedWithMessage("unable to decode LEB128 at offset 0x00000003: "
                          "uleb128 too big for uint64"));
    // The code, tag, and child byte were extracted correctly.
    EXPECT_EQ(Offset, 3u);
  }
}

TEST(DWARFDebugAbbrevTest, DWARFAbbreviationDeclSetFormExtractionError) {
  SmallString<64> RawData;
  const uint32_t Code = 1;
  const dwarf::Tag Tag = DW_TAG_compile_unit;
  const uint8_t Children = DW_CHILDREN_yes;
  const dwarf::Attribute Attr = DW_AT_name;

  // Check for malformed ULEB128.
  {
    raw_svector_ostream OS(RawData);
    encodeULEB128(Code, OS);
    encodeULEB128(Tag, OS);
    OS << Children;
    encodeULEB128(Attr, OS);
    writeMalformedULEB128Value(OS);

    uint64_t Offset = 0;
    DataExtractor Data(RawData, sys::IsLittleEndianHost, sizeof(uint64_t));
    DWARFAbbreviationDeclarationSet AbbrevSet;
    ASSERT_THAT_ERROR(
        AbbrevSet.extract(Data, &Offset),
        FailedWithMessage("unable to decode LEB128 at offset 0x00000004: "
                          "malformed uleb128, extends past end"));
    // The code, tag, child byte, and first attribute were extracted correctly.
    EXPECT_EQ(Offset, 4u);
  }

  RawData.clear();
  // Check for ULEB128 too large to fit into a uint64_t.
  {
    raw_svector_ostream OS(RawData);
    encodeULEB128(Code, OS);
    encodeULEB128(Tag, OS);
    OS << Children;
    encodeULEB128(Attr, OS);
    writeULEB128LargerThan64Bits(OS);

    uint64_t Offset = 0;
    DataExtractor Data(RawData, sys::IsLittleEndianHost, sizeof(uint64_t));
    DWARFAbbreviationDeclarationSet AbbrevSet;
    ASSERT_THAT_ERROR(
        AbbrevSet.extract(Data, &Offset),
        FailedWithMessage("unable to decode LEB128 at offset 0x00000004: "
                          "uleb128 too big for uint64"));
    // The code, tag, child byte, and first attribute were extracted correctly.
    EXPECT_EQ(Offset, 4u);
  }
}

TEST(DWARFDebugAbbrevTest, DWARFAbbrevDeclSetInvalidTag) {
  SmallString<64> RawData;
  raw_svector_ostream OS(RawData);
  uint32_t FirstCode = 1;
  // First, we're going to manually add good data.
  writeValidAbbreviationDeclarations(OS, FirstCode, InOrder);

  // Afterwards, we're going to write an Abbreviation Decl manually with an
  // invalid tag.
  encodeULEB128(FirstCode + 2, OS);
  encodeULEB128(0, OS); // Invalid Tag
  OS << static_cast<uint8_t>(DW_CHILDREN_no);
  encodeULEB128(DW_AT_name, OS);
  encodeULEB128(DW_FORM_strp, OS);
  encodeULEB128(0, OS);
  encodeULEB128(0, OS);

  encodeULEB128(0, OS);
  uint64_t Offset = 0;
  DataExtractor Data(RawData, sys::IsLittleEndianHost, sizeof(uint64_t));
  DWARFAbbreviationDeclarationSet AbbrevSet;
  EXPECT_THAT_ERROR(
      AbbrevSet.extract(Data, &Offset),
      FailedWithMessage("abbreviation declaration requires a non-null tag"));
}

TEST(DWARFDebugAbbrevTest, DWARFAbbrevDeclSetInvalidAttrValidForm) {
  SmallString<64> RawData;
  raw_svector_ostream OS(RawData);
  uint32_t FirstCode = 120;
  // First, we're going to manually add good data.
  writeValidAbbreviationDeclarations(OS, FirstCode, InOrder);

  // Afterwards, we're going to write an Abbreviation Decl manually with an
  // invalid attribute but valid form.
  encodeULEB128(FirstCode - 5, OS);
  encodeULEB128(DW_TAG_compile_unit, OS);
  OS << static_cast<uint8_t>(DW_CHILDREN_no);
  encodeULEB128(0, OS); // Invalid attribute followed by an invalid form.
  encodeULEB128(DW_FORM_strp, OS);
  encodeULEB128(0, OS);
  encodeULEB128(0, OS);

  encodeULEB128(0, OS);

  uint64_t Offset = 0;
  DataExtractor Data(RawData, sys::IsLittleEndianHost, sizeof(uint64_t));
  DWARFAbbreviationDeclarationSet AbbrevSet;
  EXPECT_THAT_ERROR(
      AbbrevSet.extract(Data, &Offset),
      FailedWithMessage(
          "malformed abbreviation declaration attribute. Either the "
          "attribute or the form is zero while the other is not"));
}

TEST(DWARFDebugAbbrevTest, DWARFAbbrevDeclSetValidAttrInvalidForm) {
  SmallString<64> RawData;
  raw_svector_ostream OS(RawData);
  uint32_t FirstCode = 120;
  // First, we're going to manually add good data.
  writeValidAbbreviationDeclarations(OS, FirstCode, InOrder);

  // Afterwards, we're going to write an Abbreviation Decl manually with a
  // valid attribute but invalid form.
  encodeULEB128(FirstCode - 5, OS);
  encodeULEB128(DW_TAG_compile_unit, OS);
  OS << static_cast<uint8_t>(DW_CHILDREN_no);
  encodeULEB128(DW_AT_name, OS);
  encodeULEB128(0, OS); // Invalid form after a valid attribute.
  encodeULEB128(0, OS);
  encodeULEB128(0, OS);

  encodeULEB128(0, OS);

  uint64_t Offset = 0;
  DataExtractor Data(RawData, sys::IsLittleEndianHost, sizeof(uint64_t));
  DWARFAbbreviationDeclarationSet AbbrevSet;
  EXPECT_THAT_ERROR(
      AbbrevSet.extract(Data, &Offset),
      FailedWithMessage(
          "malformed abbreviation declaration attribute. Either the "
          "attribute or the form is zero while the other is not"));
}

TEST(DWARFDebugAbbrevTest, DWARFAbbrevDeclSetMissingTerminator) {
  SmallString<64> RawData;
  raw_svector_ostream OS(RawData);
  uint32_t FirstCode = 120;
  // First, we're going to manually add good data.
  writeValidAbbreviationDeclarations(OS, FirstCode, InOrder);

  // Afterwards, we're going to write an Abbreviation Decl manually without a
  // termintating sequence.
  encodeULEB128(FirstCode + 7, OS);
  encodeULEB128(DW_TAG_compile_unit, OS);
  OS << static_cast<uint8_t>(DW_CHILDREN_no);
  encodeULEB128(DW_AT_name, OS);
  encodeULEB128(DW_FORM_strp, OS);

  uint64_t Offset = 0;
  DataExtractor Data(RawData, sys::IsLittleEndianHost, sizeof(uint64_t));
  DWARFAbbreviationDeclarationSet AbbrevSet;
  EXPECT_THAT_ERROR(
      AbbrevSet.extract(Data, &Offset),
      FailedWithMessage(
          "abbreviation declaration attribute list was not terminated with a "
          "null entry"));
}
