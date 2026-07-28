//===-- TestCompressedSection.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/ObjectFile/PECOFF/ObjectFilePECOFF.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/Section.h"
#include "lldb/Utility/DataExtractor.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

#if LLVM_ENABLE_ZLIB
TEST(ObjectFilePECOFFTest, GNUCompressedSection) {
  SubsystemRAII<FileSystem, ObjectFilePECOFF> subsystems;
  llvm::Expected<TestFile> file = TestFile::fromYaml(R"(
--- !COFF
OptionalHeader:
  SectionAlignment: 4096
  FileAlignment:   512
header:
  Machine:         IMAGE_FILE_MACHINE_AMD64
  Characteristics: [ IMAGE_FILE_EXECUTABLE_IMAGE, IMAGE_FILE_LARGE_ADDRESS_AWARE ]
sections:
  - Name:            .zdebug_line
    VirtualSize:     23
    Characteristics: [ IMAGE_SCN_CNT_INITIALIZED_DATA, IMAGE_SCN_MEM_READ ]
    SectionData:     5A4C49420000000000000003789C4B4C4A0600024D0127
symbols:         []
...
)");
  ASSERT_THAT_EXPECTED(file, llvm::Succeeded());

  ModuleSP module = std::make_shared<Module>(file->moduleSpec());
  ObjectFile *object = module->GetObjectFile();
  ASSERT_TRUE(llvm::isa<ObjectFilePECOFF>(object));

  SectionSP line = object->GetSectionList()->FindSectionByType(
      eSectionTypeDWARFDebugLine, true);
  ASSERT_TRUE(line);
  EXPECT_EQ(line->GetName(), ConstString(".zdebug_line"));

  DataExtractor data;
  EXPECT_EQ(line->GetSectionData(data), 3u);
  ASSERT_EQ(data.GetByteSize(), 3u);
  EXPECT_EQ(
      llvm::StringRef(reinterpret_cast<const char *>(data.GetDataStart()), 3),
      "abc");

  char tail[2];
  EXPECT_EQ(object->ReadSectionData(line.get(), 1, tail, sizeof(tail)), 2u);
  EXPECT_EQ(llvm::StringRef(tail, sizeof(tail)), "bc");
}
#endif
