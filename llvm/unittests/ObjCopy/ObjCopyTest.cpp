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

const char *SimpleFileCOFFYAML = R"(
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
)";

const char *SimpleFileELFYAML = R"(
--- !ELF
FileHeader:
  Class:    ELFCLASS64
  Data:     ELFDATA2LSB
  Type:     ET_REL
Sections:
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC ]
    Content:        "12345678"
)";

const char *SimpleFileMachOYAML = R"(
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
)";

const char *SimpleFileWasmYAML = R"(
--- !WASM
FileHeader:
  Version:         0x00000001
Sections:
  - Type: CUSTOM
    Name: text
    Payload: ABC123
...
)";

// Create ObjectFile from \p YamlCreationString and do validation using \p
// IsValidFormat checker. \p Storage is a storage for data. \returns created
// ObjectFile.
Expected<std::unique_ptr<ObjectFile>> createObjectFileFromYamlDescription(
    const char *YamlCreationString, SmallVector<char> &Storage,
    function_ref<bool(const Binary &File)> IsValidFormat) {
  auto ErrHandler = [&](const Twine &Msg) { FAIL() << "Error: " << Msg; };

  std::unique_ptr<ObjectFile> Obj =
      yaml2ObjectFile(Storage, YamlCreationString, ErrHandler);
  if (!Obj)
    return createError("could not create ObjectFile from yaml description");

  if (!IsValidFormat(*Obj))
    return createError("wrong file format");

  return std::move(Obj);
}

// Call objcopy::executeObjcopyOnBinary for \p Config and \p In. \p DataVector
// is a holder for data. \returns Binary for copied data.
Expected<std::unique_ptr<Binary>>
callObjCopy(ConfigManager &Config, object::Binary &In,
            SmallVector<char> &DataVector,
            function_ref<bool(const Binary &File)> IsValidFormat) {
  raw_svector_ostream OutStream(DataVector);

  if (Error Err = objcopy::executeObjcopyOnBinary(Config, In, OutStream))
    return std::move(Err);

  MemoryBufferRef Buffer(StringRef(DataVector.data(), DataVector.size()),
                         Config.Common.OutputFilename);

  Expected<std::unique_ptr<Binary>> Result = createBinary(Buffer);

  // Check copied file.
  if (!Result)
    return Result;

  if (!IsValidFormat(**Result))
    return createError("wrong file format");

  if (!(*Result)->isObject())
    return createError("binary is not object file");

  return Result;
}

// \returns true if specified \p File has a section named \p SectionName.
bool hasSection(ObjectFile &File, StringRef SectionName) {
  for (const object::SectionRef &Sec : File.sections()) {
    Expected<StringRef> CurSecNameOrErr = Sec.getName();
    if (!CurSecNameOrErr)
      continue;

    if (*CurSecNameOrErr == SectionName)
      return true;
  }

  return false;
}

// Check that specified \p File has a section \p SectionName and its data
// matches \p SectionData.
void checkSectionData(ObjectFile &File, StringRef SectionName,
                      StringRef SectionData) {
  for (const object::SectionRef &Sec : File.sections()) {
    Expected<StringRef> CurSecNameOrErr = Sec.getName();
    ASSERT_THAT_EXPECTED(CurSecNameOrErr, Succeeded());

    if (*CurSecNameOrErr == SectionName) {
      Expected<StringRef> CurSectionData = Sec.getContents();
      ASSERT_THAT_EXPECTED(CurSectionData, Succeeded());
      EXPECT_TRUE(Sec.getSize() == SectionData.size());
      EXPECT_TRUE(memcmp(CurSectionData->data(), SectionData.data(),
                         SectionData.size()) == 0);
      return;
    }
  }

  // Section SectionName must be presented.
  EXPECT_TRUE(false);
}

void copySimpleInMemoryFileImpl(
    const char *YamlCreationString,
    function_ref<bool(const Binary &File)> IsValidFormat) {
  SCOPED_TRACE("copySimpleInMemoryFileImpl");

  // Create Object file from YAML description.
  SmallVector<char> Storage;
  Expected<std::unique_ptr<ObjectFile>> Obj =
      createObjectFileFromYamlDescription(YamlCreationString, Storage,
                                          IsValidFormat);
  ASSERT_THAT_EXPECTED(Obj, Succeeded());

  ConfigManager Config;
  Config.Common.OutputFilename = "a.out";

  // Call executeObjcopyOnBinary()
  SmallVector<char> DataVector;
  Expected<std::unique_ptr<Binary>> Result =
      callObjCopy(Config, *Obj.get(), DataVector, IsValidFormat);
  ASSERT_THAT_EXPECTED(Result, Succeeded());
}

TEST(CopySimpleInMemoryFile, COFF) {
  SCOPED_TRACE("CopySimpleInMemoryFileCOFF");

  copySimpleInMemoryFileImpl(SimpleFileCOFFYAML,
                             [](const Binary &File) { return File.isCOFF(); });
}

TEST(CopySimpleInMemoryFile, ELF) {
  SCOPED_TRACE("CopySimpleInMemoryFileELF");

  copySimpleInMemoryFileImpl(SimpleFileELFYAML,
                             [](const Binary &File) { return File.isELF(); });
}

TEST(CopySimpleInMemoryFile, MachO) {
  SCOPED_TRACE("CopySimpleInMemoryFileMachO");

  copySimpleInMemoryFileImpl(SimpleFileMachOYAML,
                             [](const Binary &File) { return File.isMachO(); });
}

TEST(CopySimpleInMemoryFile, Wasm) {
  SCOPED_TRACE("CopySimpleInMemoryFileWasm");

  copySimpleInMemoryFileImpl(SimpleFileWasmYAML,
                             [](const Binary &File) { return File.isWasm(); });
}

enum Action : uint8_t { AddSection, UpdateSection };

void addOrUpdateSectionToFileImpl(
    const char *YamlCreationString,
    function_ref<bool(const Binary &File)> IsValidFormat,
    StringRef NewSectionName, StringRef NewSectionData, Action SectionAction) {
  SCOPED_TRACE("addOrUpdateSectionToFileImpl");

  // Create Object file from YAML description.
  SmallVector<char> Storage;
  Expected<std::unique_ptr<ObjectFile>> Obj =
      createObjectFileFromYamlDescription(YamlCreationString, Storage,
                                          IsValidFormat);
  ASSERT_THAT_EXPECTED(Obj, Succeeded());

  std::unique_ptr<MemoryBuffer> NewSectionBuffer =
      MemoryBuffer::getMemBuffer(NewSectionData, NewSectionName, false);
  std::string Name;
  if ((*Obj)->isMachO())
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
  Expected<std::unique_ptr<Binary>> Result =
      callObjCopy(Config, *Obj.get(), DataVector, IsValidFormat);
  ASSERT_THAT_EXPECTED(Result, Succeeded());

  // Check that copied file has the new section.
  checkSectionData(*static_cast<ObjectFile *>((*Result).get()), NewSectionName,
                   NewSectionData);
}

TEST(AddSection, COFF) {
  SCOPED_TRACE("addSectionToFileCOFF");

  addOrUpdateSectionToFileImpl(
      SimpleFileCOFFYAML, [](const Binary &File) { return File.isCOFF(); },
      ".foo", "1234", AddSection);
}

TEST(AddSection, ELF) {
  SCOPED_TRACE("addSectionToFileELF");

  addOrUpdateSectionToFileImpl(
      SimpleFileELFYAML, [](const Binary &File) { return File.isELF(); },
      ".foo", "1234", AddSection);
}

TEST(AddSection, MachO) {
  SCOPED_TRACE("addSectionToFileMachO");

  addOrUpdateSectionToFileImpl(
      SimpleFileMachOYAML, [](const Binary &File) { return File.isMachO(); },
      "__foo", "1234", AddSection);
}

TEST(AddSection, Wasm) {
  SCOPED_TRACE("addSectionToFileWasm");

  addOrUpdateSectionToFileImpl(
      SimpleFileWasmYAML, [](const Binary &File) { return File.isWasm(); },
      ".foo", "1234", AddSection);
}

TEST(UpdateSection, COFF) {
  SCOPED_TRACE("updateSectionToFileCOFF");

  addOrUpdateSectionToFileImpl(
      SimpleFileCOFFYAML, [](const Binary &File) { return File.isCOFF(); },
      ".text", "1234", UpdateSection);
}

TEST(UpdateSection, ELF) {
  SCOPED_TRACE("updateSectionToFileELF");

  addOrUpdateSectionToFileImpl(
      SimpleFileELFYAML, [](const Binary &File) { return File.isELF(); },
      ".text", "1234", UpdateSection);
}

TEST(UpdateSection, MachO) {
  SCOPED_TRACE("updateSectionToFileMachO");

  addOrUpdateSectionToFileImpl(
      SimpleFileMachOYAML, [](const Binary &File) { return File.isMachO(); },
      "__text", "1234", UpdateSection);
}

void removeSectionByPatternImpl(
    const char *YamlCreationString,
    function_ref<bool(const Binary &File)> IsValidFormat,
    StringRef SectionWildcard, StringRef SectionName) {
  SCOPED_TRACE("removeSectionByPatternImpl");

  // Create Object file from YAML description.
  SmallVector<char> Storage;
  Expected<std::unique_ptr<ObjectFile>> Obj =
      createObjectFileFromYamlDescription(YamlCreationString, Storage,
                                          IsValidFormat);
  ASSERT_THAT_EXPECTED(Obj, Succeeded());

  // Check that section is present.
  EXPECT_TRUE(hasSection(**Obj, SectionName));

  Expected<NameOrPattern> Pattern = objcopy::NameOrPattern::create(
      SectionWildcard, objcopy::MatchStyle::Wildcard,
      [](Error Err) { return Err; });

  ConfigManager Config;
  Config.Common.OutputFilename = "a.out";
  EXPECT_THAT_ERROR(Config.Common.ToRemove.addMatcher(std::move(Pattern)),
                    Succeeded());

  SmallVector<char> DataVector;
  Expected<std::unique_ptr<Binary>> Result =
      callObjCopy(Config, *Obj.get(), DataVector, IsValidFormat);
  ASSERT_THAT_EXPECTED(Result, Succeeded());

  // Check that section was removed.
  EXPECT_FALSE(
      hasSection(*static_cast<ObjectFile *>((*Result).get()), SectionName));
}

TEST(RemoveSectionByPattern, COFF) {
  SCOPED_TRACE("removeSectionByPatternCOFF");

  removeSectionByPatternImpl(
      SimpleFileCOFFYAML, [](const Binary &File) { return File.isCOFF(); },
      "\\.text*", ".text");
}

TEST(RemoveSectionByPattern, ELF) {
  SCOPED_TRACE("removeSectionByPatternELF");

  removeSectionByPatternImpl(
      SimpleFileELFYAML, [](const Binary &File) { return File.isELF(); },
      "\\.text*", ".text");
}

TEST(RemoveSectionByPattern, MachO) {
  SCOPED_TRACE("removeSectionByPatternMachO");

  removeSectionByPatternImpl(
      SimpleFileMachOYAML, [](const Binary &File) { return File.isMachO(); },
      "__TEXT,__text*", "__text");
}

TEST(RemoveSectionByPattern, Wasm) {
  SCOPED_TRACE("removeSectionByPatternWasm");

  removeSectionByPatternImpl(
      SimpleFileWasmYAML, [](const Binary &File) { return File.isWasm(); },
      "text*", "text");
}
