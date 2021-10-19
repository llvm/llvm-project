//===-- XcodeSDKModuleTests.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Platform/MacOSX/PlatformMacOSX.h"
#include "Plugins/SymbolFile/DWARF/DWARFCompileUnit.h"
#include "Plugins/SymbolFile/DWARF/DWARFDIE.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "TestingSupport/Symbol/YAMLModuleTester.h"
#include "lldb/Core/PluginManager.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

#ifdef __APPLE__
namespace {
class XcodeSDKModuleTests : public testing::Test {
  SubsystemRAII<HostInfoBase, PlatformMacOSX> subsystems;
};
} // namespace


TEST_F(XcodeSDKModuleTests, TestModuleGetXcodeSDK) {
  const char *yamldata = R"(
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_EXEC
  Machine: EM_386
DWARF:
  debug_str:
    - MacOSX10.9.sdk
  debug_abbrev:
    - Table:
        - Code:            0x00000001
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_language
              Form:            DW_FORM_data2
            - Attribute:       DW_AT_APPLE_sdk
              Form:            DW_FORM_strp
  debug_info:
    - Version:         2
      AddrSize:        8
      Entries:
        - AbbrCode:        0x00000001
          Values:
            - Value:           0x000000000000000C
            - Value:           0x0000000000000000
        - AbbrCode:        0x00000000
...
)";

  YAMLModuleTester t(yamldata);
  DWARFUnit *dwarf_unit = t.GetDwarfUnit();
  auto *dwarf_cu = llvm::cast<DWARFCompileUnit>(dwarf_unit);
  ASSERT_TRUE(static_cast<bool>(dwarf_cu));
  SymbolFileDWARF &sym_file = dwarf_cu->GetSymbolFileDWARF();
  CompUnitSP comp_unit = sym_file.GetCompileUnitAtIndex(0);
  ASSERT_TRUE(static_cast<bool>(comp_unit.get()));
  ModuleSP module = t.GetModule();
  ASSERT_EQ(module->GetSourceMappingList().GetSize(), 0u);
  XcodeSDK sdk = sym_file.ParseXcodeSDK(*comp_unit);
  ASSERT_EQ(sdk.GetType(), XcodeSDK::Type::MacOSX);
  ASSERT_EQ(module->GetSourceMappingList().GetSize(), 1u);
}
#endif
