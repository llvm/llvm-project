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
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

#ifdef __APPLE__
namespace {
class XcodeSDKModuleTests : public testing::Test {
  SubsystemRAII<HostInfoBase, PlatformMacOSX> subsystems;
};

struct SDKPathParsingTestData {
  /// Each path will be put into a new CU's
  /// DW_AT_LLVM_sysroot.
  std::vector<llvm::StringRef> input_sdk_paths;

  /// 'true' if we expect \ref GetSDKPathFromDebugInfo
  /// to notify us about an SDK mismatch.
  bool expect_mismatch;

  /// 'true if the test expects the parsed SDK to
  /// be an internal one.
  bool expect_internal_sdk;

  /// A substring that the final parsed sdk
  /// is expected to contain.
  llvm::StringRef expect_sdk_path_pattern;
};

struct SDKPathParsingMultiparamTests
    : public XcodeSDKModuleTests,
      public testing::WithParamInterface<SDKPathParsingTestData> {
  std::vector<std::string>
  createCompileUnits(std::vector<llvm::StringRef> const &sdk_paths) {
    std::vector<std::string> compile_units;

    for (auto sdk_path : sdk_paths) {
      compile_units.emplace_back(llvm::formatv(
          R"(
        - Version:         2
          AddrSize:        8
          AbbrevTableID:   0
          AbbrOffset:      0x0
          Entries:
            - AbbrCode:        0x00000001
              Values:
                - Value:       0x000000000000000C
                - CStr:        {0}
                - CStr:        {1}
            - AbbrCode:        0x00000000
            )",
          llvm::sys::path::filename(sdk_path, llvm::sys::path::Style::posix),
          sdk_path));
    }

    return compile_units;
  }
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

TEST_F(XcodeSDKModuleTests, TestSDKPathFromDebugInfo_InvalidSDKPath) {
  // Tests that parsing a CU with an invalid SDK directory name fails.

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
            - Attribute:       DW_AT_language
              Form:            DW_FORM_data2
            - Attribute:       DW_AT_APPLE_sdk
              Form:            DW_FORM_string
  debug_info:
    - Version:         2
      AddrSize:        8
      AbbrevTableID:   0
      AbbrOffset:      0x0
      Entries:
        - AbbrCode:        0x00000001
          Values:
            - Value:       0x000000000000000C
            - CStr:        "1abc@defgh2"
        - AbbrCode:        0x00000000
...
)";

  YAMLModuleTester t(yamldata);
  ModuleSP module = t.GetModule();
  ASSERT_NE(module, nullptr);

  auto path_or_err = PlatformDarwin::ResolveSDKPathFromDebugInfo(*module);
  EXPECT_FALSE(static_cast<bool>(path_or_err));
  llvm::consumeError(path_or_err.takeError());
}

TEST_F(XcodeSDKModuleTests, TestSDKPathFromDebugInfo_No_DW_AT_APPLE_sdk) {
  // Tests that parsing a CU without a DW_AT_APPLE_sdk fails.

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
            - Attribute:       DW_AT_language
              Form:            DW_FORM_data2
            - Attribute:       DW_AT_LLVM_sysroot
              Form:            DW_FORM_string
  debug_info:
    - Version:         2
      AddrSize:        8
      AbbrevTableID:   0
      AbbrOffset:      0x0
      Entries:
        - AbbrCode:        0x00000001
          Values:
            - Value:       0x000000000000000C
            - CStr:        "/Library/Developer/CommandLineTools/SDKs/iPhoneOS14.0.Internal.sdk"
        - AbbrCode:        0x00000000
...
)";

  YAMLModuleTester t(yamldata);
  ModuleSP module = t.GetModule();
  ASSERT_NE(module, nullptr);

  auto path_or_err = PlatformDarwin::ResolveSDKPathFromDebugInfo(*module);
  EXPECT_FALSE(static_cast<bool>(path_or_err));
  llvm::consumeError(path_or_err.takeError());
}

TEST_P(SDKPathParsingMultiparamTests, TestSDKPathFromDebugInfo) {
  // Tests that we can parse the SDK path from debug-info.
  // In the presence of multiple compile units, one of which
  // points to an internal SDK, we should pick the internal SDK.

  std::string yamldata = R"(
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
            - Attribute:       DW_AT_language
              Form:            DW_FORM_data2
            - Attribute:       DW_AT_APPLE_sdk
              Form:            DW_FORM_string
            - Attribute:       DW_AT_LLVM_sysroot
              Form:            DW_FORM_string
  debug_info:
)";

  auto [input_sdk_paths, expect_mismatch, expect_internal_sdk,
        expect_sdk_path_pattern] = GetParam();

  for (auto &&sdk : createCompileUnits(input_sdk_paths))
    yamldata += std::move(sdk);

  YAMLModuleTester t(yamldata);
  DWARFUnit *dwarf_unit = t.GetDwarfUnit();
  auto *dwarf_cu = llvm::cast<DWARFCompileUnit>(dwarf_unit);
  ASSERT_TRUE(static_cast<bool>(dwarf_cu));
  SymbolFileDWARF &sym_file = dwarf_cu->GetSymbolFileDWARF();
  ASSERT_EQ(sym_file.GetNumCompileUnits(), input_sdk_paths.size());
  ModuleSP module = t.GetModule();
  ASSERT_NE(module, nullptr);

  auto sdk_or_err = PlatformDarwin::GetSDKPathFromDebugInfo(*module);
  ASSERT_TRUE(static_cast<bool>(sdk_or_err));

  auto [sdk, found_mismatch] = *sdk_or_err;

  EXPECT_EQ(found_mismatch, expect_mismatch);
  EXPECT_EQ(sdk.IsAppleInternalSDK(), expect_internal_sdk);
  EXPECT_NE(sdk.GetString().find(expect_sdk_path_pattern), std::string::npos);
}

SDKPathParsingTestData sdkPathParsingTestCases[] = {
    /// Multiple CUs with a mix of internal and public SDKs
    {.input_sdk_paths =
         {"/Library/Developer/CommandLineTools/SDKs/MacOSX10.9.sdk",
          "/invalid/path/to/something.invalid.sdk",
          "/Library/Developer/CommandLineTools/SDKs/iPhoneOS14.0.Internal.sdk",
          "/Library/Developer/CommandLineTools/SDKs/MacOSX10.9.sdk"},
     .expect_mismatch = true,
     .expect_internal_sdk = true,
     .expect_sdk_path_pattern = "Internal.sdk"},

    /// Single CU with a public SDK
    {.input_sdk_paths =
         {"/Library/Developer/CommandLineTools/SDKs/MacOSX10.9.sdk"},
     .expect_mismatch = false,
     .expect_internal_sdk = false,
     .expect_sdk_path_pattern = "MacOSX10.9.sdk"},

    /// Single CU with an internal SDK
    {.input_sdk_paths =
         {"/Library/Developer/CommandLineTools/SDKs/iPhoneOS14.0.Internal.sdk"},
     .expect_mismatch = false,
     .expect_internal_sdk = true,
     .expect_sdk_path_pattern = "Internal.sdk"},

    /// Two CUs with an internal SDK each
    {.input_sdk_paths =
         {"/Library/Developer/CommandLineTools/SDKs/iPhoneOS14.0.Internal.sdk",
          "/Library/Developer/CommandLineTools/SDKs/iPhoneOS12.9.Internal.sdk"},
     .expect_mismatch = false,
     .expect_internal_sdk = true,
     .expect_sdk_path_pattern = "Internal.sdk"},

    /// Two CUs with an internal SDK each
    {.input_sdk_paths =
         {"/Library/Developer/CommandLineTools/SDKs/iPhoneOS14.1.sdk",
          "/Library/Developer/CommandLineTools/SDKs/MacOSX11.3.sdk"},
     .expect_mismatch = false,
     .expect_internal_sdk = false,
     .expect_sdk_path_pattern = "iPhoneOS14.1.sdk"},
};

INSTANTIATE_TEST_CASE_P(SDKPathParsingTests, SDKPathParsingMultiparamTests,
                        ::testing::ValuesIn(sdkPathParsingTestCases));
#endif
