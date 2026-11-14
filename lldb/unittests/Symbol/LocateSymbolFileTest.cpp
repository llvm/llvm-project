//===-- SymbolsTest.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "Plugins/SymbolLocator/Default/SymbolLocatorDefault.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Target.h"
#include "llvm/Support/FileUtilities.h"

using namespace lldb_private;

namespace {
class SymbolsTest : public ::testing::Test {
public:
  SubsystemRAII<FileSystem, HostInfo> subsystems;
};
} // namespace

TEST_F(
    SymbolsTest,
    TerminateLocateExecutableSymbolFileForUnknownExecutableAndUnknownSymbolFile) {
  ModuleSpec module_spec;
  FileSpecList search_paths = Target::GetDefaultDebugFileSearchPaths();
  StatisticsMap map;
  FileSpec symbol_file_spec =
      PluginManager::LocateExecutableSymbolFile(module_spec, search_paths, map);
  EXPECT_TRUE(symbol_file_spec.GetFilename().empty());
}

TEST_F(SymbolsTest,
       LocateExecutableSymbolFileForUnknownExecutableAndMissingSymbolFile) {
  ModuleSpec module_spec;
  // using a GUID here because the symbol file shouldn't actually exist on disk
  module_spec.GetSymbolFileSpec().SetFile(
      "4A524676-B24B-4F4E-968A-551D465EBAF1.so", FileSpec::Style::native);
  FileSpecList search_paths = Target::GetDefaultDebugFileSearchPaths();
  StatisticsMap map;
  FileSpec symbol_file_spec =
      PluginManager::LocateExecutableSymbolFile(module_spec, search_paths, map);
  EXPECT_TRUE(symbol_file_spec.GetFilename().empty());
}

TEST_F(SymbolsTest, LocateEmbeddedObjectUsingItsDataAndBounds) {
  SubsystemRAII<ObjectFileELF> objects;
  auto ExpectedFile = TestFile::fromYaml(R"(
--- !ELF
FileHeader:
  Class: ELFCLASS64
  Data: ELFDATA2LSB
  Type: ET_REL
  Machine: EM_X86_64
Sections:
  - Name: .text
    Type: SHT_PROGBITS
    Flags: [ SHF_ALLOC, SHF_EXECINSTR ]
    Content: "90"
...
)");
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());
  ModuleSpec original = ExpectedFile->moduleSpec();
  original.GetFileSpec() =
      FileSpec("/nonexistent-module-spec-test/container.core");
  original.SetObjectOffset(0x10000);
  auto module = std::make_shared<Module>(original);
  ASSERT_NE(module->GetObjectFile(), nullptr);
  ModuleSpec spec = module->GetModuleSpec();
  auto found = SymbolLocatorDefault::LocateExecutableObjectFile(spec);
  ASSERT_TRUE(found.has_value());
  EXPECT_EQ(found->GetFileSpec(), spec.GetFileSpec());

  const uint8_t wrong_uuid[] = {1, 2, 3, 4, 5, 6, 7, 8};
  spec.GetUUID() = UUID(wrong_uuid, sizeof(wrong_uuid));
  EXPECT_FALSE(SymbolLocatorDefault::LocateExecutableObjectFile(spec));
}

TEST_F(SymbolsTest, FileBackedModuleSpecDoesNotMaterializeObjectBytes) {
  SubsystemRAII<ObjectFileELF> objects;
  auto ExpectedFile = TestFile::fromYaml(R"(
--- !ELF
FileHeader:
  Class: ELFCLASS64
  Data: ELFDATA2LSB
  Type: ET_REL
  Machine: EM_X86_64
Sections:
  - Name: .text
    Type: SHT_PROGBITS
    Flags: [ SHF_ALLOC, SHF_EXECINSTR ]
    Content: "90"
...
)");
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());
  llvm::Expected<llvm::sys::fs::TempFile> temp_file =
      ExpectedFile->writeToTemporaryFile();
  ASSERT_THAT_EXPECTED(temp_file, llvm::Succeeded());
  std::string path = temp_file->TmpName;
  llvm::FileRemover file_remover(path);
  ASSERT_THAT_ERROR(temp_file->keep(), llvm::Succeeded());

  auto module = std::make_shared<Module>(ModuleSpec(FileSpec(path)));
  ASSERT_NE(module->GetObjectFile(), nullptr);

  ModuleSpec spec = module->GetModuleSpec();
  EXPECT_EQ(spec.GetExtractor(), nullptr);
  EXPECT_EQ(spec.GetObjectOffset(), module->GetObjectOffset());
  EXPECT_EQ(spec.GetObjectSize(), module->GetObjectSize());
}
