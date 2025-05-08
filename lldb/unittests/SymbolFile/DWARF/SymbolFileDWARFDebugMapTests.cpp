//===-- SymbolFileDWARFDebugMapTests.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/SymbolFile/DWARF/SymbolFileDWARFDebugMap.h"
#include "TestingSupport/TestUtilities.h"

#include "lldb/Core/Module.h"
#include "llvm/Testing/Support/Error.h"

#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::plugin::dwarf;

TEST(SymbolFileDWARFDebugMapTests, CreateInstanceReturnNullForNonMachOFile) {
  // Make sure we don't crash parsing a null unit DIE.
  const char *yamldata = R"(
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_EXEC
  Machine: EM_386
DWARF:
  debug_abbrev:
    - Table:
        - Code:            0x00000001
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_addr_base
              Form:            DW_FORM_sec_offset
  debug_info:
    - Version:         5
      AddrSize:        4
      UnitType:        DW_UT_compile
      Entries:
        - AbbrCode:        0x00000001
          Values:
            - Value:           0x8 # Offset of the first Address past the header
        - AbbrCode:        0x0
  debug_addr:
    - Version: 5
      AddressSize: 4
      Entries:
        - Address: 0x1234
        - Address: 0x5678
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

  // bool waiting = true;
  // while (waiting) {}

  llvm::Expected<TestFile> file = TestFile::fromYaml(yamldata);
  EXPECT_THAT_EXPECTED(file, llvm::Succeeded());
  auto module_sp = std::make_shared<Module>(file->moduleSpec());
  auto object_file = module_sp->GetObjectFile();
  ASSERT_NE(object_file, nullptr);

  auto debug_map = SymbolFileDWARFDebugMap::CreateInstance(object_file->shared_from_this());
  ASSERT_EQ(debug_map, nullptr);
}
