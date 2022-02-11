//===- ObjCopyTest.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ObjCopy/ObjCopy.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ObjCopy/ConfigManager.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/ObjectYAML/yaml2obj.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace object;
using namespace objcopy;
using namespace yaml;

void copySimpleInMemoryFileImpl(
    const char *YamlCreationString,
    std::function<bool(const Binary &File)> IsValidFormat) {
  auto ErrHandler = [&](const Twine &Msg) { FAIL() << "Error: " << Msg; };

  // Create Object file from YAML description.
  SmallVector<char> Storage;
  std::unique_ptr<ObjectFile> Obj =
      yaml2ObjectFile(Storage, YamlCreationString, ErrHandler);
  ASSERT_TRUE(Obj);
  ASSERT_TRUE(IsValidFormat(*Obj));

  ConfigManager Config;
  Config.Common.OutputFilename = "a.out";

  // Call executeObjcopyOnBinary()
  SmallVector<char> DataVector;
  raw_svector_ostream OutStream(DataVector);
  Error Err = objcopy::executeObjcopyOnBinary(Config, *Obj.get(), OutStream);
  ASSERT_FALSE(std::move(Err));

  MemoryBufferRef Buffer(StringRef(DataVector.data(), DataVector.size()),
                         Config.Common.OutputFilename);

  // Check copied file.
  Expected<std::unique_ptr<Binary>> Result = createBinary(Buffer);
  ASSERT_THAT_EXPECTED(Result, Succeeded());
  ASSERT_TRUE(IsValidFormat(**Result));
}

TEST(CopySimpleInMemoryFile, COFF) {
  SCOPED_TRACE("CopySimpleInMemoryFileCOFF");

  copySimpleInMemoryFileImpl(
      R"(
--- !COFF
header:
  Machine:         IMAGE_FILE_MACHINE_AMD64
  Characteristics: [  ]
sections:
  - Name:            .text
    Characteristics: [  ]
    Alignment:       4
    SectionData:     E800000000C3C3C3
symbols:
...
)",
      [](const Binary &File) { return File.isCOFF(); });
}

TEST(CopySimpleInMemoryFile, ELF) {
  SCOPED_TRACE("CopySimpleInMemoryFileELF");

  copySimpleInMemoryFileImpl(
      R"(
--- !ELF
FileHeader:
   Class:    ELFCLASS64
   Data:     ELFDATA2LSB
   Type:     ET_REL)",
      [](const Binary &File) { return File.isELF(); });
}

TEST(CopySimpleInMemoryFile, MachO) {
  SCOPED_TRACE("CopySimpleInMemoryFileMachO");

  copySimpleInMemoryFileImpl(
      R"(
--- !mach-o
FileHeader:
  magic:           0xFEEDFACF
  cputype:         0x01000007
  cpusubtype:      0x80000003
  filetype:        0x00000002
  ncmds:           0
  sizeofcmds:      0
  flags:           0x00218085
  reserved:        0x00000000
...
)",
      [](const Binary &File) { return File.isMachO(); });
}

TEST(CopySimpleInMemoryFile, Wasm) {
  SCOPED_TRACE("CopySimpleInMemoryFileWasm");

  copySimpleInMemoryFileImpl(
      R"(
--- !WASM
FileHeader:
  Version:         0x00000001
...
)",
      [](const Binary &File) { return File.isWasm(); });
}
