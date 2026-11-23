//===- BuildIDTest.cpp - Tests for getBuildID ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/BuildID.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/ObjectYAML/yaml2obj.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Testing/Support/Error.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::object;

template <class ELFT>
static Expected<ELFObjectFile<ELFT>> toBinary(SmallVectorImpl<char> &Storage,
                                              StringRef Yaml) {
  raw_svector_ostream OS(Storage);
  yaml::Input YIn(Yaml);
  if (!yaml::convertYAML(YIn, OS, [](const Twine &Msg) {}))
    return createStringError(std::errc::invalid_argument,
                             "unable to convert YAML");
  return ELFObjectFile<ELFT>::create(MemoryBufferRef(OS.str(), "dummyELF"));
}

static StringRef getInvalidNoteELF(bool WithShdr) {
  static std::string WithSection(R"(
--- !ELF
FileHeader:
  Class:          ELFCLASS64
  Data:           ELFDATA2LSB
  Type:           ET_EXEC
  Machine:        EM_X86_64
ProgramHeaders:
  - Type:         PT_NOTE
    FileSize:     0x1a
    FirstSec:     .note.gnu.build-id
    LastSec:      .note.gnu.build-id
Sections:
  - Name:         .note.gnu.build-id
    Type:         SHT_NOTE
    AddressAlign: 0x04
    Notes:
      - Name:     "GNU"
        Desc:     "abb50d82b6bdc861"
        Type:     3
)");
  static std::string WithoutSection(WithSection + R"(
  - Type:         SectionHeaderTable
    NoHeaders:    true
)");
  if (WithShdr)
    return WithSection;
  return WithoutSection;
}

// The BuildID can be looked up from a section header, if there is no program
// header.
TEST(BuildIDTest, InvalidPhdrFileSizeWithShdrs) {
  SmallString<0> Storage;
  Expected<ELFObjectFile<ELF64LE>> ElfOrErr =
      toBinary<ELF64LE>(Storage, getInvalidNoteELF(true));
  ASSERT_THAT_EXPECTED(ElfOrErr, Succeeded());
  BuildIDRef BuildID = getBuildID(&ElfOrErr.get());
  EXPECT_EQ(
      StringRef(reinterpret_cast<const char *>(BuildID.data()), BuildID.size()),
      "\xAB\xB5\x0D\x82\xB6\xBD\xC8\x61");
}

// The code handles a malformed program header that points at data outside the
// file.
TEST(BuildIDTest, InvalidPhdrFileSizeNoShdrs) {
  SmallString<0> Storage;
  Expected<ELFObjectFile<ELF64LE>> ElfOrErr =
      toBinary<ELF64LE>(Storage, getInvalidNoteELF(false));
  ASSERT_THAT_EXPECTED(ElfOrErr, Succeeded());
  BuildIDRef BuildID = getBuildID(&ElfOrErr.get());
  EXPECT_EQ(
      StringRef(reinterpret_cast<const char *>(BuildID.data()), BuildID.size()),
      "");
}

// The code handles a malformed section header that points at data outside the
// file.
TEST(BuildIDTest, InvalidSectionHeader) {
  SmallString<0> Storage;
  Expected<ELFObjectFile<ELF64LE>> ElfOrErr = toBinary<ELF64LE>(Storage, R"(
--- !ELF
FileHeader:
  Class:          ELFCLASS64
  Data:           ELFDATA2LSB
  Type:           ET_EXEC
  Machine:        EM_X86_64
ProgramHeaders:
  - Type:         PT_NOTE
    FirstSec:     .note.gnu.build-id
    LastSec:      .note.gnu.build-id
Sections:
  - Name:         .note.gnu.build-id
    Type:         SHT_NOTE
    AddressAlign: 0x04
    ShOffset:     0x1a1
    Notes:
      - Name:     "GNU"
        Desc:     "abb50d82b6bdc861"
        Type:     3
)");
  ASSERT_THAT_EXPECTED(ElfOrErr, Succeeded());
  BuildIDRef BuildID = getBuildID(&ElfOrErr.get());
  EXPECT_EQ(
      StringRef(reinterpret_cast<const char *>(BuildID.data()), BuildID.size()),
      "\xAB\xB5\x0D\x82\xB6\xBD\xC8\x61");
}
