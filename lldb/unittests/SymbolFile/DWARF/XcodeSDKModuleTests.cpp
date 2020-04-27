//===-- XcodeSDKModuleTests.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/SymbolFile/DWARF/DWARFASTParserClang.h"
#include "Plugins/SymbolFile/DWARF/DWARFCompileUnit.h"
#include "Plugins/SymbolFile/DWARF/DWARFDIE.h"
#include "TestingSupport/Symbol/YAMLModuleTester.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

#ifndef __APPLE__
namespace {
class XcodeSDKModuleTests : public testing::Test {
  void SetUp() override { PlatformDarwin::Initialize(); }
  void TearDown() override { PlatformDarwin::Terminate(); }
};
} // namespace


TEST_F(XcodeSDKModuleTests, TestModuleGetXcodeSDK) {
  const char *yamldata = R"(
debug_str:
  - MacOSX10.9.sdk
debug_abbrev:
  - Code:            0x00000001
    Tag:             DW_TAG_compile_unit
    Children:        DW_CHILDREN_no
    Attributes:
      - Attribute:       DW_AT_language
        Form:            DW_FORM_data2
      - Attribute:       DW_AT_APPLE_sdk
        Form:            DW_FORM_strp
debug_info:
  - Length:
      TotalLength:     8
    Version:         2
    AbbrOffset:      0
    AddrSize:        8
    Entries:
      - AbbrCode:        0x00000001
        Values:
          - Value:           0x000000000000000C
          - Value:           0x0000000000000000
      - AbbrCode:        0x00000000
        Values:          []
...
)";

  YAMLModuleTester t(yamldata, "x86_64-apple-macosx");
  auto dwarf_unit_sp = t.GetDwarfUnit();
  auto *dwarf_cu = llvm::cast<DWARFCompileUnit>(dwarf_unit_sp.get());
  ASSERT_TRUE((bool)dwarf_cu);
  ASSERT_TRUE((bool)dwarf_cu->GetSymbolFileDWARF().GetCompUnitForDWARFCompUnit(
      *dwarf_cu));
  auto module = t.GetModule();
  XcodeSDK sdk = module->GetXcodeSDK();
  ASSERT_EQ(sdk.GetType(), XcodeSDK::Type::MacOSX);
}
#endif
