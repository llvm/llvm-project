//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "Plugins/ObjectFile/PECOFF/ObjectFilePECOFF.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"

#include "lldb/Core/Module.h"
#include "lldb/Core/Section.h"
#include "lldb/Symbol/Symtab.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "llvm/Testing/Support/Error.h"

using namespace lldb_private;
using namespace lldb;

class MemoryInstanceTest : public testing::Test {
  SubsystemRAII<FileSystem, ObjectFilePECOFF> subsystems;
};

// Process::ReadModuleFromMemory reads 512 bytes of an image by default. For
// this DLL that covers the headers but not the export table they point to, so
// the object file is created without a COFF binary.
TEST_F(MemoryInstanceTest, HeadersOnly) {
  llvm::Expected<TestFile> file = TestFile::fromYaml(R"(
--- !COFF
OptionalHeader:
  ImageBase:       0x180000000
  SectionAlignment: 4096
  FileAlignment:   512
  ExportTable:
    RelativeVirtualAddress: 4096
    Size:            40
header:
  Machine:         IMAGE_FILE_MACHINE_AMD64
  Characteristics: [ IMAGE_FILE_EXECUTABLE_IMAGE, IMAGE_FILE_LARGE_ADDRESS_AWARE, IMAGE_FILE_DLL ]
sections:
  - Name:            .rdata
    VirtualAddress:  4096
    VirtualSize:     40
    SizeOfRawData:   512
    Characteristics: [ IMAGE_SCN_CNT_INITIALIZED_DATA, IMAGE_SCN_MEM_READ ]
    SectionData:     00000000000000000000000000000000000000000000000000000000000000000000000000000000
symbols:         []
...
)");
  ASSERT_THAT_EXPECTED(file, llvm::Succeeded());

  DataExtractorSP file_data = file->moduleSpec().GetExtractor();
  ASSERT_GT(file_data->GetByteSize(), 512u);
  auto header_data =
      std::make_shared<DataBufferHeap>(file_data->GetDataStart(), 512);

  auto module_sp = std::make_shared<Module>(ModuleSpec());
  std::unique_ptr<ObjectFile> object_file(
      ObjectFilePECOFF::CreateMemoryInstance(module_sp, header_data,
                                             ProcessSP(), 0x180000000));
  ASSERT_NE(object_file, nullptr);
  auto *pecoff = static_cast<ObjectFilePECOFF *>(object_file.get());

  Symtab *symtab = object_file->GetSymtab();
  ASSERT_NE(symtab, nullptr);
  EXPECT_EQ(symtab->GetNumSymbols(), 0u);
  EXPECT_FALSE(pecoff->GetDebugLink());
  EXPECT_FALSE(pecoff->GetPDBPath());

  SectionList *section_list = object_file->GetSectionList();
  ASSERT_NE(section_list, nullptr);
  SectionSP rdata = section_list->FindSectionByName(ConstString(".rdata"));
  ASSERT_NE(rdata, nullptr);
  DataExtractor section_data;
  EXPECT_EQ(object_file->ReadSectionData(rdata.get(), section_data), 0u);
}
