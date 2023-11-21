//===- llvm/unittest/DebugInfo/DWARFDieTest.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DWARF/DWARFCompileUnit.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/ObjectYAML/DWARFEmitter.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::dwarf;

namespace {

TEST(DWARFDie, getLocations) {
  const char *yamldata = R"(
    debug_abbrev:
      - Table:
          - Code:            0x00000001
            Tag:             DW_TAG_compile_unit
            Children:        DW_CHILDREN_no
            Attributes:
              - Attribute:       DW_AT_location
                Form:            DW_FORM_sec_offset
              - Attribute:       DW_AT_data_member_location
                Form:            DW_FORM_exprloc
              - Attribute:       DW_AT_vtable_elem_location
                Form:            DW_FORM_sec_offset
              - Attribute:       DW_AT_call_data_location
                Form:            DW_FORM_sec_offset
    debug_info:
      - Version:         5
        UnitType:        DW_UT_compile
        AddrSize:        4
        Entries:
          - AbbrCode:        0x00000001
            Values:
              - Value:           12
              - Value:           0x0000000000000001
                BlockData:       [ 0x47 ]
              - Value:           20
              - Value:           25
    debug_loclists:
      - AddressSize:      4
        OffsetEntryCount: 0
        Lists:
          - Entries:
              - Operator: DW_LLE_start_length
                Values:   [ 0x01, 0x02 ]
              - Operator: DW_LLE_end_of_list
          - Entries:
              - Operator: DW_LLE_startx_length
                Values:   [ 0x01, 0x02 ]
              - Operator: DW_LLE_end_of_list
          - Entries:
              - Operator: DW_LLE_start_length
                Values:   [ 0x01, 0x02 ]
              ## end_of_list intentionally missing.
  )";
  Expected<StringMap<std::unique_ptr<MemoryBuffer>>> Sections =
      DWARFYAML::emitDebugSections(StringRef(yamldata),
                                   /*IsLittleEndian=*/true,
                                   /*Is64BitAddrSize=*/false);
  ASSERT_THAT_EXPECTED(Sections, Succeeded());
  std::unique_ptr<DWARFContext> Ctx =
      DWARFContext::create(*Sections, 4, /*isLittleEndian=*/true);
  DWARFCompileUnit *CU = Ctx->getCompileUnitForOffset(0);
  ASSERT_NE(nullptr, CU);
  DWARFDie Die = CU->getUnitDIE();
  ASSERT_TRUE(Die.isValid());

  EXPECT_THAT_EXPECTED(Die.getLocations(DW_AT_location),
                       HasValue(testing::ElementsAre(DWARFLocationExpression{
                           DWARFAddressRange{1, 3}, {}})));

  EXPECT_THAT_EXPECTED(Die.getLocations(DW_AT_data_member_location),
                       HasValue(testing::ElementsAre(
                           DWARFLocationExpression{std::nullopt, {0x47}})));

  EXPECT_THAT_EXPECTED(
      Die.getLocations(DW_AT_vtable_elem_location),
      Failed<ErrorInfoBase>(testing::Property(
          &ErrorInfoBase::message,
          "unable to resolve indirect address 1 for: DW_LLE_startx_length")));

  EXPECT_THAT_EXPECTED(
      Die.getLocations(DW_AT_call_data_location),
      FailedWithMessage(
          "unexpected end of data at offset 0x20 while reading [0x20, 0x21)"));

  EXPECT_THAT_EXPECTED(
      Die.getLocations(DW_AT_call_data_value),
      Failed<ErrorInfoBase>(testing::Property(&ErrorInfoBase::message,
                                              "No DW_AT_call_data_value")));
}

TEST(DWARFDie, getDeclFile) {
  const char *yamldata = R"(
  debug_str:
    - ''
  debug_abbrev:
    - ID:              0
      Table:
        - Code:            0x1
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_stmt_list
              Form:            DW_FORM_sec_offset
        - Code:            0x2
          Tag:             DW_TAG_subprogram
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_decl_file
              Form:            DW_FORM_data1
  debug_info:
    - Length:          0xF
      Version:         4
      AbbrevTableID:   0
      AbbrOffset:      0x0
      AddrSize:        8
      Entries:
        - AbbrCode:        0x1
          Values:
            - Value:           0x0
        - AbbrCode:        0x2
          Values:
            - Value:           0x1
        - AbbrCode:        0x0
  debug_line:
    - Length:          42
      Version:         2
      PrologueLength:  36
      MinInstLength:   1
      DefaultIsStmt:   1
      LineBase:        251
      LineRange:       14
      OpcodeBase:      13
      StandardOpcodeLengths: [ 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1 ]
      IncludeDirs:
        - '/tmp'
      Files:
        - Name:            main.cpp
          DirIdx:          1
          ModTime:         0
          Length:          0
  )";

  // Given DWARF like this:
  //
  // 0x0000000b: DW_TAG_compile_unit
  //               DW_AT_stmt_list (0x00000000)
  //
  // 0x00000010:   DW_TAG_subprogram
  //                 DW_AT_decl_file ("/tmp/main.cpp")
  //
  // 0x00000012:   NULL
  //
  // This tests that we can extract the right DW_AT_decl_file from a DIE that
  // has a DW_AT_decl_file attribute.

  Expected<StringMap<std::unique_ptr<MemoryBuffer>>> Sections =
      DWARFYAML::emitDebugSections(StringRef(yamldata),
                                   /*IsLittleEndian=*/true,
                                   /*Is64BitAddrSize=*/true);
  ASSERT_THAT_EXPECTED(Sections, Succeeded());
  std::unique_ptr<DWARFContext> Ctx =
      DWARFContext::create(*Sections, 4, /*isLittleEndian=*/true);
  DWARFCompileUnit *CU = Ctx->getCompileUnitForOffset(0);
  ASSERT_NE(nullptr, CU);
  DWARFDie Die = CU->getUnitDIE(/*ExtractUnitDIEOnly=*/false);
  ASSERT_TRUE(Die.isValid());

  DWARFDie MainDie = Die.getFirstChild();
  ASSERT_TRUE(MainDie.isValid());

  std::string DeclFile = MainDie.getDeclFile(
      DILineInfoSpecifier::FileLineInfoKind::AbsoluteFilePath);

  std::string Ref =
      ("/tmp" + llvm::sys::path::get_separator() + "main.cpp").str();
  EXPECT_EQ(DeclFile, Ref);
}

TEST(DWARFDie, getDeclFileAbstractOrigin) {
  const char *yamldata = R"(
  debug_str:
    - ''
  debug_abbrev:
    - ID:              0
      Table:
        - Code:            0x1
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_stmt_list
              Form:            DW_FORM_sec_offset
        - Code:            0x2
          Tag:             DW_TAG_subprogram
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_abstract_origin
              Form:            DW_FORM_ref_addr
        - Code:            0x3
          Tag:             DW_TAG_subprogram
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_decl_file
              Form:            DW_FORM_data1
  debug_info:
    - Length:          0x14
      Version:         4
      AbbrevTableID:   0
      AbbrOffset:      0x0
      AddrSize:        8
      Entries:
        - AbbrCode:        0x1
          Values:
            - Value:           0x0
        - AbbrCode:        0x2
          Values:
            - Value:           0x15
        - AbbrCode:        0x3
          Values:
            - Value:           0x1
        - AbbrCode:        0x0
  debug_line:
    - Length:          42
      Version:         2
      PrologueLength:  36
      MinInstLength:   1
      DefaultIsStmt:   1
      LineBase:        251
      LineRange:       14
      OpcodeBase:      13
      StandardOpcodeLengths: [ 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1 ]
      IncludeDirs:
        - '/tmp'
      Files:
        - Name:            main.cpp
          DirIdx:          1
          ModTime:         0
          Length:          0
  )";

  // Given DWARF like this:
  //
  // 0x0000000b: DW_TAG_compile_unit
  //               DW_AT_stmt_list (0x00000000)
  //
  // 0x00000010:   DW_TAG_subprogram
  //                 DW_AT_abstract_origin (0x0000000000000015)
  //
  // 0x00000015:   DW_TAG_subprogram
  //                 DW_AT_decl_file ("/tmp/main.cpp")
  //
  // 0x00000017:   NULL
  //
  //
  // The DIE at 0x00000010 uses a DW_AT_abstract_origin to point to the DIE at
  // 0x00000015, make sure that DWARFDie::getDeclFile() succeeds by extracting
  // the right file name of "/tmp/main.cpp".
  //
  // This tests that when we have a DW_AT_abstract_origin with a compile unit
  // relative form (DW_FORM_ref4) to another DIE that we get the right
  // DW_AT_decl_file value.

  Expected<StringMap<std::unique_ptr<MemoryBuffer>>> Sections =
      DWARFYAML::emitDebugSections(StringRef(yamldata),
                                   /*IsLittleEndian=*/true,
                                   /*Is64BitAddrSize=*/true);
  ASSERT_THAT_EXPECTED(Sections, Succeeded());
  std::unique_ptr<DWARFContext> Ctx =
      DWARFContext::create(*Sections, 4, /*isLittleEndian=*/true);
  DWARFCompileUnit *CU = Ctx->getCompileUnitForOffset(0);
  ASSERT_NE(nullptr, CU);
  DWARFDie Die = CU->getUnitDIE(/*ExtractUnitDIEOnly=*/false);
  ASSERT_TRUE(Die.isValid());

  DWARFDie MainDie = Die.getFirstChild();
  ASSERT_TRUE(MainDie.isValid());

  std::string DeclFile = MainDie.getDeclFile(
      DILineInfoSpecifier::FileLineInfoKind::AbsoluteFilePath);

  std::string Ref =
      ("/tmp" + llvm::sys::path::get_separator() + "main.cpp").str();
  EXPECT_EQ(DeclFile, Ref);
}

TEST(DWARFDie, getDeclFileSpecification) {
  const char *yamldata = R"(
  debug_str:
    - ''
  debug_abbrev:
    - ID:              0
      Table:
        - Code:            0x1
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_stmt_list
              Form:            DW_FORM_sec_offset
        - Code:            0x2
          Tag:             DW_TAG_subprogram
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_specification
              Form:            DW_FORM_ref_addr
        - Code:            0x3
          Tag:             DW_TAG_subprogram
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_decl_file
              Form:            DW_FORM_data1
  debug_info:
    - Length:          0x14
      Version:         4
      AbbrevTableID:   0
      AbbrOffset:      0x0
      AddrSize:        8
      Entries:
        - AbbrCode:        0x1
          Values:
            - Value:           0x0
        - AbbrCode:        0x2
          Values:
            - Value:           0x15
        - AbbrCode:        0x3
          Values:
            - Value:           0x1
        - AbbrCode:        0x0
  debug_line:
    - Length:          42
      Version:         2
      PrologueLength:  36
      MinInstLength:   1
      DefaultIsStmt:   1
      LineBase:        251
      LineRange:       14
      OpcodeBase:      13
      StandardOpcodeLengths: [ 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1 ]
      IncludeDirs:
        - '/tmp'
      Files:
        - Name:            main.cpp
          DirIdx:          1
          ModTime:         0
          Length:          0
  )";

  // Given DWARF like this:
  //
  // 0x0000000b: DW_TAG_compile_unit
  //               DW_AT_stmt_list   (0x00000000)
  //
  // 0x00000010:   DW_TAG_subprogram
  //                 DW_AT_specification     (0x0000000000000015)
  //
  // 0x00000015:   DW_TAG_subprogram
  //                 DW_AT_decl_file ("/tmp/main.cpp")
  //
  // 0x00000017:   NULL
  //
  // The DIE at 0x00000010 uses a DW_AT_specification to point to the DIE at
  // 0x00000015, make sure that DWARFDie::getDeclFile() succeeds by extracting
  // the right file name of "/tmp/main.cpp".
  //
  // This tests that when we have a DW_AT_specification with a compile unit
  // relative form (DW_FORM_ref4) to another DIE that we get the right
  // DW_AT_decl_file value.

  Expected<StringMap<std::unique_ptr<MemoryBuffer>>> Sections =
      DWARFYAML::emitDebugSections(StringRef(yamldata),
                                   /*IsLittleEndian=*/true,
                                   /*Is64BitAddrSize=*/true);
  ASSERT_THAT_EXPECTED(Sections, Succeeded());
  std::unique_ptr<DWARFContext> Ctx =
      DWARFContext::create(*Sections, 4, /*isLittleEndian=*/true);
  DWARFCompileUnit *CU = Ctx->getCompileUnitForOffset(0);
  ASSERT_NE(nullptr, CU);
  DWARFDie Die = CU->getUnitDIE(/*ExtractUnitDIEOnly=*/false);
  ASSERT_TRUE(Die.isValid());

  DWARFDie MainDie = Die.getFirstChild();
  ASSERT_TRUE(MainDie.isValid());

  std::string DeclFile = MainDie.getDeclFile(
      DILineInfoSpecifier::FileLineInfoKind::AbsoluteFilePath);

  std::string Ref =
      ("/tmp" + llvm::sys::path::get_separator() + "main.cpp").str();
  EXPECT_EQ(DeclFile, Ref);
}

TEST(DWARFDie, getDeclFileAbstractOriginAcrossCUBoundary) {
  const char *yamldata = R"(
  debug_str:
    - ''
  debug_abbrev:
    - ID:              0
      Table:
        - Code:            0x1
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
        - Code:            0x2
          Tag:             DW_TAG_subprogram
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_abstract_origin
              Form:            DW_FORM_ref_addr
        - Code:            0x3
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_stmt_list
              Form:            DW_FORM_sec_offset
        - Code:            0x4
          Tag:             DW_TAG_subprogram
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_decl_file
              Form:            DW_FORM_data1
  debug_info:
    - Length:          0xE
      Version:         4
      AbbrevTableID:   0
      AbbrOffset:      0x0
      AddrSize:        8
      Entries:
        - AbbrCode:        0x1
        - AbbrCode:        0x2
          Values:
            - Value:           0x22
        - AbbrCode:        0x0
    - Length:          0xF
      Version:         4
      AbbrevTableID:   0
      AbbrOffset:      0x0
      AddrSize:        8
      Entries:
        - AbbrCode:        0x3
          Values:
            - Value:           0x0
        - AbbrCode:        0x4
          Values:
            - Value:           0x1
        - AbbrCode:        0x0
  debug_line:
    - Length:          42
      Version:         2
      PrologueLength:  36
      MinInstLength:   1
      DefaultIsStmt:   1
      LineBase:        251
      LineRange:       14
      OpcodeBase:      13
      StandardOpcodeLengths: [ 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1 ]
      IncludeDirs:
        - '/tmp'
      Files:
        - Name:            main.cpp
          DirIdx:          1
          ModTime:         0
          Length:          0
  )";

  // Given DWARF like this:
  //
  // 0x0000000b: DW_TAG_compile_unit
  //
  // 0x0000000c:   DW_TAG_subprogram
  //                 DW_AT_abstract_origin (0x0000000000000022)
  //
  // 0x00000011:   NULL
  //
  // 0x0000001d: DW_TAG_compile_unit
  //               DW_AT_stmt_list (0x00000000)
  //
  // 0x00000022:   DW_TAG_subprogram
  //                 DW_AT_decl_file ("/tmp/main.cpp")
  //
  // 0x00000024:   NULL
  //
  // This tests that when we have a DW_AT_abstract_origin with a
  // DW_FORM_ref_addr to another DIE in another compile unit that we use the
  // right file table when converting the file index of the DW_AT_decl_file.
  //
  // The DIE at 0x0000000c uses a DW_AT_abstract_origin to point to the DIE at
  // 0x00000022, make sure that DWARFDie::getDeclFile() succeeds by extracting
  // the right file name of "/tmp/main.cpp". The DW_AT_decl_file must grab the
  // file from the line table prologue of the compile unit at offset
  // 0x0000001d.

  Expected<StringMap<std::unique_ptr<MemoryBuffer>>> Sections =
      DWARFYAML::emitDebugSections(StringRef(yamldata),
                                   /*IsLittleEndian=*/true,
                                   /*Is64BitAddrSize=*/true);
  ASSERT_THAT_EXPECTED(Sections, Succeeded());
  std::unique_ptr<DWARFContext> Ctx =
      DWARFContext::create(*Sections, 4, /*isLittleEndian=*/true);
  DWARFCompileUnit *CU = Ctx->getCompileUnitForOffset(0);
  ASSERT_NE(nullptr, CU);
  DWARFDie Die = CU->getUnitDIE(/*ExtractUnitDIEOnly=*/false);
  ASSERT_TRUE(Die.isValid());

  DWARFDie MainDie = Die.getFirstChild();
  ASSERT_TRUE(MainDie.isValid());

  std::string DeclFile = MainDie.getDeclFile(
      DILineInfoSpecifier::FileLineInfoKind::AbsoluteFilePath);

  std::string Ref =
      ("/tmp" + llvm::sys::path::get_separator() + "main.cpp").str();
  EXPECT_EQ(DeclFile, Ref);
}

TEST(DWARFDie, getDeclFileSpecificationAcrossCUBoundary) {
  const char *yamldata = R"(
  debug_str:
    - ''
  debug_abbrev:
    - ID:              0
      Table:
        - Code:            0x1
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
        - Code:            0x2
          Tag:             DW_TAG_subprogram
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_specification
              Form:            DW_FORM_ref_addr
        - Code:            0x3
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_stmt_list
              Form:            DW_FORM_sec_offset
        - Code:            0x4
          Tag:             DW_TAG_subprogram
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_decl_file
              Form:            DW_FORM_data1
  debug_info:
    - Length:          0xE
      Version:         4
      AbbrevTableID:   0
      AbbrOffset:      0x0
      AddrSize:        8
      Entries:
        - AbbrCode:        0x1
        - AbbrCode:        0x2
          Values:
            - Value:           0x22
        - AbbrCode:        0x0
    - Length:          0xF
      Version:         4
      AbbrevTableID:   0
      AbbrOffset:      0x0
      AddrSize:        8
      Entries:
        - AbbrCode:        0x3
          Values:
            - Value:           0x0
        - AbbrCode:        0x4
          Values:
            - Value:           0x1
        - AbbrCode:        0x0
  debug_line:
    - Length:          42
      Version:         2
      PrologueLength:  36
      MinInstLength:   1
      DefaultIsStmt:   1
      LineBase:        251
      LineRange:       14
      OpcodeBase:      13
      StandardOpcodeLengths: [ 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1 ]
      IncludeDirs:
        - '/tmp'
      Files:
        - Name:            main.cpp
          DirIdx:          1
          ModTime:         0
          Length:          0
  )";

  // Given DWARF like this:
  //
  // 0x0000000b: DW_TAG_compile_unit
  //
  // 0x0000000c:   DW_TAG_subprogram
  //                 DW_AT_specification     (0x0000000000000022)
  //
  // 0x00000011:   NULL
  //
  // 0x0000001d: DW_TAG_compile_unit
  //               DW_AT_stmt_list   (0x00000000)
  //
  // 0x00000022:   DW_TAG_subprogram
  //                 DW_AT_decl_file ("/tmp/main.cpp")
  //
  // 0x00000024:   NULL
  //
  // This tests that when we have a DW_AT_specification with a
  // DW_FORM_ref_addr to another DIE in another compile unit that we use the
  // right file table when converting the file index of the DW_AT_decl_file.
  //
  // The DIE at 0x0000000c uses a DW_AT_specification to point to the DIE at
  // 0x00000022, make sure that DWARFDie::getDeclFile() succeeds by extracting
  // the right file name of "/tmp/main.cpp". The DW_AT_decl_file must grab the
  // file from the line table prologue of the compile unit at offset
  // 0x0000001d.

  Expected<StringMap<std::unique_ptr<MemoryBuffer>>> Sections =
      DWARFYAML::emitDebugSections(StringRef(yamldata),
                                   /*IsLittleEndian=*/true,
                                   /*Is64BitAddrSize=*/true);
  ASSERT_THAT_EXPECTED(Sections, Succeeded());
  std::unique_ptr<DWARFContext> Ctx =
      DWARFContext::create(*Sections, 4, /*isLittleEndian=*/true);
  DWARFCompileUnit *CU = Ctx->getCompileUnitForOffset(0);
  ASSERT_NE(nullptr, CU);
  DWARFDie Die = CU->getUnitDIE(/*ExtractUnitDIEOnly=*/false);
  ASSERT_TRUE(Die.isValid());

  DWARFDie MainDie = Die.getFirstChild();
  ASSERT_TRUE(MainDie.isValid());

  std::string DeclFile = MainDie.getDeclFile(
      DILineInfoSpecifier::FileLineInfoKind::AbsoluteFilePath);

  std::string Ref =
      ("/tmp" + llvm::sys::path::get_separator() + "main.cpp").str();
  EXPECT_EQ(DeclFile, Ref);
}

} // end anonymous namespace
