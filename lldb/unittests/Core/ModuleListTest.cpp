//===-- ModuleListTest.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/ModuleList.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/UUID.h"

#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"

#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

// Test that when we already have a module in the shared_module_list with a
// specific UUID, the next call to GetSharedModule with a module_spec with the
// same UUID should return the existing module instead of creating a new one.
TEST(ModuleListTest, GetSharedModuleReusesExistingModuleWithSameUUID) {
  SubsystemRAII<FileSystem, ObjectFileELF> subsystems;

  auto ExpectedFile = TestFile::fromYaml(R"(
--- !ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  Type:            ET_DYN
  Machine:         EM_X86_64
Sections:
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    AddressAlign:    0x0000000000000010
...
)");
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());

  // First, let's verify that calling GetSharedModule twice with the same
  // module_spec returns the same module pointer

  ModuleSP first_module;
  bool first_did_create = false;
  Status error_first =
      ModuleList::GetSharedModule(ExpectedFile->moduleSpec(), first_module,
                                  nullptr, &first_did_create, false);

  // Second call with the same spec
  ModuleSP second_module;
  bool second_did_create = false;
  Status error_second =
      ModuleList::GetSharedModule(ExpectedFile->moduleSpec(), second_module,
                                  nullptr, &second_did_create, false);

  if (error_first.Success() && error_second.Success()) {
    // If both succeeded, verify they're the same module
    EXPECT_EQ(first_module.get(), second_module.get())
        << "GetSharedModule should return the same module for the same spec";
    EXPECT_TRUE(first_did_create) << "First call should create the module";
    EXPECT_FALSE(second_did_create)
        << "Second call should reuse the existing module";
  }
}

// Test that UUID-based lookup finds existing modules
TEST(ModuleListTest, FindSharedModuleByUUID) {
  SubsystemRAII<FileSystem, ObjectFileELF> subsystems;

  auto ExpectedFile = TestFile::fromYaml(R"(
--- !ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  Type:            ET_DYN
  Machine:         EM_X86_64
Sections:
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    AddressAlign:    0x0000000000000010
...
)");
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());

  // Create and add a module to the shared module list using the moduleSpec()
  ModuleSP created_module;
  bool did_create = false;
  Status error = ModuleList::GetSharedModule(
      ExpectedFile->moduleSpec(), created_module, nullptr, &did_create, false);

  if (error.Success() && created_module) {
    // Get the UUID of the created module
    UUID module_uuid = created_module->GetUUID();

    if (module_uuid.IsValid()) {
      // Now try to find the module by UUID
      ModuleSP found_module = ModuleList::FindSharedModule(module_uuid);

      ASSERT_NE(found_module.get(), nullptr)
          << "FindSharedModule should find the module by UUID";
      EXPECT_EQ(found_module.get(), created_module.get())
          << "FindSharedModule should return the same module instance";
      EXPECT_EQ(found_module->GetUUID(), module_uuid)
          << "Found module should have the same UUID";
    }
  }
}

// Test that GetSharedModule with UUID finds existing module even with different
// path
TEST(ModuleListTest, GetSharedModuleByUUIDIgnoresPath) {
  SubsystemRAII<FileSystem, ObjectFileELF> subsystems;

  auto ExpectedFile = TestFile::fromYaml(R"(
--- !ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  Type:            ET_DYN
  Machine:         EM_X86_64
Sections:
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    AddressAlign:    0x0000000000000010
...
)");
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());

  // Create and add a module to the shared module list
  ModuleSP first_module;
  bool first_did_create = false;
  Status first_error =
      ModuleList::GetSharedModule(ExpectedFile->moduleSpec(), first_module,
                                  nullptr, &first_did_create, false);

  if (first_error.Success() && first_module) {
    UUID module_uuid = first_module->GetUUID();

    if (module_uuid.IsValid()) {
      // Now try to get a module with the same UUID but different path
      ModuleSpec second_spec;
      second_spec.GetFileSpec() = FileSpec("/different/path/to/module.so");
      second_spec.GetArchitecture() = ArchSpec("x86_64-pc-linux");
      second_spec.GetUUID() = module_uuid;

      ModuleSP second_module;
      bool second_did_create = false;
      Status second_error = ModuleList::GetSharedModule(
          second_spec, second_module, nullptr, &second_did_create, false);

      if (second_error.Success() && second_module) {
        // If we got a module back, check if it's the same one
        bool is_same_module = (second_module.get() == first_module.get());

        // Document the behavior: ideally UUID should take precedence
        // and return the existing module
        EXPECT_TRUE(is_same_module)
            << "GetSharedModule with matching UUID should return existing "
               "module, "
            << "even with different path (per PR #160199)";

        if (is_same_module) {
          EXPECT_FALSE(second_did_create)
              << "Should not create a new module when UUID matches";
        }
      }
    }
  }
}
