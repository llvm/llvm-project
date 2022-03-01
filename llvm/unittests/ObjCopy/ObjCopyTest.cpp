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

enum Action : uint8_t { AddSection, UpdateSection };

void addOrUpdateSectionToFileImpl(
    const char *YamlCreationString,
    std::function<bool(const Binary &File)> IsValidFormat,
    StringRef NewSectionName, StringRef NewSectionData, Action SectionAction) {
  auto ErrHandler = [&](const Twine &Msg) { FAIL() << "Error: " << Msg; };

  // Create Object file from YAML description.
  SmallVector<char> Storage;
  std::unique_ptr<ObjectFile> Obj =
      yaml2ObjectFile(Storage, YamlCreationString, ErrHandler);
  ASSERT_TRUE(Obj);
  ASSERT_TRUE(IsValidFormat(*Obj));

  std::unique_ptr<MemoryBuffer> NewSectionBuffer =
      MemoryBuffer::getMemBuffer(NewSectionData, NewSectionName, false);
  std::string Name;
  if (Obj->isMachO())
    Name = "__TEXT," + NewSectionName.str();
  else
    Name = NewSectionName.str();

  ConfigManager Config;
  Config.Common.OutputFilename = "a.out";
  if (SectionAction == AddSection)
    Config.Common.AddSection.push_back({Name, std::move(NewSectionBuffer)});
  else
    Config.Common.UpdateSection.push_back({Name, std::move(NewSectionBuffer)});

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
  ASSERT_TRUE((*Result)->isObject());

  // Check that copied file has the new section.
  bool HasNewSection = false;
  for (const object::SectionRef &Sect :
       static_cast<ObjectFile *>((*Result).get())->sections()) {
    Expected<StringRef> SectNameOrErr = Sect.getName();
    ASSERT_THAT_EXPECTED(SectNameOrErr, Succeeded());

    if (*SectNameOrErr == NewSectionName) {
      HasNewSection = true;
      Expected<StringRef> SectionData = Sect.getContents();
      ASSERT_THAT_EXPECTED(SectionData, Succeeded());
      EXPECT_TRUE(Sect.getSize() == NewSectionData.size());
      EXPECT_TRUE(memcmp(SectionData->data(), NewSectionData.data(),
                         NewSectionData.size()) == 0);
      break;
    }
  }
  EXPECT_TRUE(HasNewSection);
}

TEST(AddSection, COFF) {
  SCOPED_TRACE("addSectionToFileCOFF");

  addOrUpdateSectionToFileImpl(
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
      [](const Binary &File) { return File.isCOFF(); }, ".foo", "1234",
      AddSection);
}

TEST(AddSection, ELF) {
  SCOPED_TRACE("addSectionToFileELF");

  addOrUpdateSectionToFileImpl(
      R"(
--- !ELF
FileHeader:
   Class:    ELFCLASS64
   Data:     ELFDATA2LSB
   Type:     ET_REL)",
      [](const Binary &File) { return File.isELF(); }, ".foo", "1234",
      AddSection);
}

TEST(AddSection, MachO) {
  SCOPED_TRACE("addSectionToFileMachO");

  addOrUpdateSectionToFileImpl(
      R"(
--- !mach-o
FileHeader:
  magic:           0xFEEDFACF
  cputype:         0x01000007
  cpusubtype:      0x80000003
  filetype:        0x00000001
  ncmds:           1
  sizeofcmds:      152
  flags:           0x00002000
  reserved:        0x00000000
LoadCommands:
  - cmd:             LC_SEGMENT_64
    cmdsize:         152
    segname:         __TEXT
    vmaddr:          0
    vmsize:          4
    fileoff:         184
    filesize:        4
    maxprot:         7
    initprot:        7
    nsects:          1
    flags:           0
    Sections:
      - sectname:        __text
        segname:         __TEXT
        addr:            0x0000000000000000
        content:         'AABBCCDD'
        size:            4
        offset:          184
        align:           0
        reloff:          0x00000000
        nreloc:          0
        flags:           0x80000400
        reserved1:       0x00000000
        reserved2:       0x00000000
        reserved3:       0x00000000
...
)",
      [](const Binary &File) { return File.isMachO(); }, "__foo", "1234",
      AddSection);
}

TEST(AddSection, Wasm) {
  SCOPED_TRACE("addSectionToFileWasm");

  addOrUpdateSectionToFileImpl(
      R"(
--- !WASM
FileHeader:
  Version:         0x00000001
...
)",
      [](const Binary &File) { return File.isWasm(); }, ".foo", "1234",
      AddSection);
}

TEST(UpdateSection, COFF) {
  SCOPED_TRACE("updateSectionToFileCOFF");

  addOrUpdateSectionToFileImpl(
      R"(
--- !COFF
header:
  Machine:         IMAGE_FILE_MACHINE_AMD64
  Characteristics: [  ]
sections:
  - Name:            .foo
    Characteristics: [  ]
    Alignment:       4
    SectionData:     E800000000C3C3C3
symbols:
...
)",
      [](const Binary &File) { return File.isCOFF(); }, ".foo", "1234",
      UpdateSection);
}

TEST(UpdateSection, ELF) {
  SCOPED_TRACE("updateSectionToFileELF");

  addOrUpdateSectionToFileImpl(
      R"(
--- !ELF
FileHeader:
  Class:    ELFCLASS64
  Data:     ELFDATA2LSB
  Type:     ET_REL
Sections:
  - Name:            .foo
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC ]
    Content:        "12345678"
)",
      [](const Binary &File) { return File.isELF(); }, ".foo", "1234",
      UpdateSection);
}

TEST(UpdateSection, MachO) {
  SCOPED_TRACE("updateSectionToFileMachO");

  addOrUpdateSectionToFileImpl(
      R"(
--- !mach-o
FileHeader:
  magic:           0xFEEDFACF
  cputype:         0x01000007
  cpusubtype:      0x80000003
  filetype:        0x00000001
  ncmds:           1
  sizeofcmds:      152
  flags:           0x00002000
  reserved:        0x00000000
LoadCommands:
  - cmd:             LC_SEGMENT_64
    cmdsize:         152
    segname:         __TEXT
    vmaddr:          0
    vmsize:          4
    fileoff:         184
    filesize:        4
    maxprot:         7
    initprot:        7
    nsects:          1
    flags:           0
    Sections:
      - sectname:        __foo
        segname:         __TEXT
        addr:            0x0000000000000000
        content:         'AABBCCDD'
        size:            4
        offset:          184
        align:           0
        reloff:          0x00000000
        nreloc:          0
        flags:           0x80000400
        reserved1:       0x00000000
        reserved2:       0x00000000
        reserved3:       0x00000000
...
)",
      [](const Binary &File) { return File.isMachO(); }, "__foo", "1234",
      UpdateSection);
}
