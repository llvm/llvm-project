//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "Plugins/ObjectFile/PECOFF/ObjectFilePECOFF.h"
#include "Plugins/SymbolFile/NativePDB/SymbolFileNativePDB.h"
#include "Plugins/SymbolFile/PDB/SymbolFilePDB.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/Module.h"
#include "lldb/Host/FileSystem.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace lldb_private;

namespace {
class PDBFactoryTest : public testing::Test {
protected:
  SubsystemRAII<FileSystem, ObjectFileELF, ObjectFilePECOFF, SymbolFilePDB>
      subsystems;
};
} // namespace

// An ELF file without DWARF ends up with SymbolFileSymtab with or without this
// check, so the selected plugin can't show the change; test the factory.
TEST_F(PDBFactoryTest, RejectsELF) {
  auto file = TestFile::fromYaml(R"(
--- !ELF
FileHeader:
  Class: ELFCLASS64
  Data: ELFDATA2LSB
  Type: ET_REL
  Machine: EM_X86_64
Sections:
  - Name: .text
    Type: SHT_PROGBITS
    Content: "90"
...
)");
  ASSERT_THAT_EXPECTED(file, llvm::Succeeded());
  auto module = std::make_shared<Module>(file->moduleSpec());
  ASSERT_NE(module->GetObjectFile(), nullptr);
  auto object = module->GetObjectFile()->shared_from_this();
  std::unique_ptr<SymbolFile> symfile(
      npdb::SymbolFileNativePDB::CreateInstance(object));
  EXPECT_EQ(symfile, nullptr);
}

TEST_F(PDBFactoryTest, AcceptsPECOFF) {
  if (!SymbolFilePDB::UseNativePDB())
    GTEST_SKIP() << "native PDB reader is not selected";

  auto file = TestFile::fromYaml(R"(
--- !COFF
OptionalHeader:
  SectionAlignment: 4096
  FileAlignment:   512
header:
  Machine:         IMAGE_FILE_MACHINE_AMD64
  Characteristics: [ IMAGE_FILE_EXECUTABLE_IMAGE, IMAGE_FILE_LARGE_ADDRESS_AWARE ]
sections:
  - Name:            .text
    VirtualSize:     1
    SizeOfRawData:   512
    Characteristics: [ IMAGE_SCN_CNT_CODE, IMAGE_SCN_MEM_EXECUTE, IMAGE_SCN_MEM_READ ]
    SectionData:     c3
symbols:         []
...
)");
  ASSERT_THAT_EXPECTED(file, llvm::Succeeded());
  auto module = std::make_shared<Module>(file->moduleSpec());
  ASSERT_NE(module->GetObjectFile(), nullptr);
  auto object = module->GetObjectFile()->shared_from_this();
  std::unique_ptr<SymbolFile> symfile(
      npdb::SymbolFileNativePDB::CreateInstance(object));
  EXPECT_NE(symfile, nullptr);
}
